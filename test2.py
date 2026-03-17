import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from net.model import AdaIR
from utils.dataset_utils import DenoiseTestDataset
from utils.image_io import save_image_tensor
from utils.image_utils import crop_img
from utils.video_prior_utils import TemporalPriorExtractor


class AdaIRModel(pl.LightningModule):
    def __init__(self, prior_strength=1.0):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.prior_strength = float(prior_strength)

    def forward(self, x, prior=None, prior_strength=None):
        strength = self.prior_strength if prior_strength is None else float(prior_strength)
        return self.net(x, noise_emb=prior, prior_strength=strength)


def _load_lightning_weights(module, ckpt_path, map_location="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = module.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print("Missing keys (showing up to 10): {}".format(missing[:10]))
    if unexpected:
        print("Unexpected keys (showing up to 10): {}".format(unexpected[:10]))
    return module


class RestoreInputOnlyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_dir,
        use_video_prior=False,
        prior_window_size=9,
        prior_align_mode="ecc",
        prior_warp_mode="affine",
        prior_temporal_semantics="stable",
        prior_stable_band_ratio=0.10,
        prior_prefilter_alpha=0.0,
        prior_prefilter_threshold=0.35,
        prior_prefilter_mask_cap=0.65,
        prior_detail_protect=0.75,
        prior_artifact_relax=0.8,
        prior_residual_boost=0.55,
        save_filter_debug=False,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.to_tensor = ToTensor()
        self.use_video_prior = use_video_prior
        self.prior_prefilter_alpha = float(prior_prefilter_alpha)
        self.prior_prefilter_threshold = float(prior_prefilter_threshold)
        self.prior_prefilter_mask_cap = float(prior_prefilter_mask_cap)
        self.prior_detail_protect = float(prior_detail_protect)
        self.prior_artifact_relax = float(prior_artifact_relax)
        self.prior_residual_boost = float(prior_residual_boost)
        self.save_filter_debug = bool(save_filter_debug)
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self.files = sorted([
            name for name in os.listdir(input_dir)
            if os.path.splitext(name)[1].lower() in valid_ext
        ])
        self.file_paths = [os.path.join(input_dir, name) for name in self.files]

        self.prior_extractor = None
        if self.use_video_prior:
            self.prior_extractor = TemporalPriorExtractor(
                frame_paths=self.file_paths,
                window_size=prior_window_size,
                align_mode=prior_align_mode,
                warp_mode=prior_warp_mode,
                temporal_semantics=prior_temporal_semantics,
                stable_band_ratio=prior_stable_band_ratio,
                crop_base=16,
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        if self.prior_extractor is None:
            file_path = os.path.join(self.input_dir, file_name)
            image = crop_img(np.array(Image.open(file_path).convert("RGB")), base=16)
            prior = None
        else:
            original_image = self.prior_extractor.get_frame(idx)
            if self.prior_prefilter_alpha > 0:
                image = self.prior_extractor.get_filtered_frame(
                    idx,
                    alpha=self.prior_prefilter_alpha,
                    threshold=self.prior_prefilter_threshold,
                    mask_cap=self.prior_prefilter_mask_cap,
                    detail_protect=self.prior_detail_protect,
                    artifact_relax=self.prior_artifact_relax,
                    residual_boost=self.prior_residual_boost,
                )
            else:
                image = original_image
            prior = self.prior_extractor.get_prior(idx)

        image = self.to_tensor(image)
        stem = os.path.splitext(file_name)[0]
        if prior is None:
            return [stem], image

        prior = torch.from_numpy(np.transpose(prior, (2, 0, 1))).float()
        if self.save_filter_debug and self.prior_prefilter_alpha > 0:
            original_image = self.to_tensor(original_image)
            return [stem], image, prior, original_image
        return [stem], image, prior


def _ensure_trailing_slash(path):
    return path if path.endswith('/') else path + '/'


def _has_image_files(path):
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path) and os.path.splitext(name)[1].lower() in valid_ext:
            return True
    return False


def _resolve_task_paths(base_path, default_splits):
    resolved_paths = []

    if os.path.isdir(os.path.join(base_path, 'input')):
        return [_ensure_trailing_slash(base_path)]
    if _has_image_files(base_path):
        return [_ensure_trailing_slash(base_path)]

    for split in default_splits:
        split_path = os.path.join(base_path, split)
        if os.path.isdir(os.path.join(split_path, 'input')):
            resolved_paths.append(_ensure_trailing_slash(split_path))
    if resolved_paths:
        return resolved_paths

    if os.path.isdir(base_path):
        for split in sorted(os.listdir(base_path)):
            split_path = os.path.join(base_path, split)
            if os.path.isdir(os.path.join(split_path, 'input')) or _has_image_files(split_path):
                resolved_paths.append(_ensure_trailing_slash(split_path))

    return resolved_paths


def _resolve_input_dir(split_path):
    input_dir = os.path.join(split_path, "input")
    return input_dir if os.path.isdir(input_dir) else split_path


def _build_restore_dataset(input_dir, args):
    return RestoreInputOnlyDataset(
        input_dir=input_dir,
        use_video_prior=args.use_video_prior,
        prior_window_size=args.prior_window_size,
        prior_align_mode=args.prior_align_mode,
        prior_warp_mode=args.prior_warp_mode,
        prior_temporal_semantics=args.prior_temporal_semantics,
        prior_stable_band_ratio=args.prior_stable_band_ratio,
        prior_prefilter_alpha=args.prior_prefilter_alpha,
        prior_prefilter_threshold=args.prior_prefilter_threshold,
        prior_prefilter_mask_cap=args.prior_prefilter_mask_cap,
        prior_detail_protect=args.prior_detail_protect,
        prior_artifact_relax=args.prior_artifact_relax,
        prior_residual_boost=args.prior_residual_boost,
        save_filter_debug=args.save_filter_debug,
    )


def infer_denoise(net, dataset, sigma, output_root, split_name, device):
    output_dir = os.path.join(output_root, "denoise", split_name, str(sigma))
    os.makedirs(output_dir, exist_ok=True)

    dataset.set_sigma(sigma)
    if len(dataset) == 0:
        raise RuntimeError(f"No denoise input images found in: {dataset.args.denoise_path}")
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    with torch.no_grad():
        for ([clean_name], degrad_patch, _) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)
            restored = net(degrad_patch)
            save_image_tensor(restored, os.path.join(output_dir, clean_name[0] + ".png"))


def _save_prior_visuals(prior_patch, prior_dir, frame_name):
    os.makedirs(prior_dir, exist_ok=True)
    save_image_tensor(prior_patch, os.path.join(prior_dir, frame_name + "_prior_rgb.png"))
    save_image_tensor(prior_patch[:, 0:1], os.path.join(prior_dir, frame_name + "_r.png"))
    save_image_tensor(prior_patch[:, 1:2], os.path.join(prior_dir, frame_name + "_g.png"))
    save_image_tensor(prior_patch[:, 2:3], os.path.join(prior_dir, frame_name + "_b.png"))


def _save_filter_debug_visuals(original_patch, filtered_patch, debug_dir, frame_name, gain=8.0):
    os.makedirs(debug_dir, exist_ok=True)
    save_image_tensor(original_patch, os.path.join(debug_dir, frame_name + "_orig.png"))
    save_image_tensor(filtered_patch, os.path.join(debug_dir, frame_name + "_filtered.png"))

    removed_abs = torch.abs(original_patch - filtered_patch)
    save_image_tensor(removed_abs, os.path.join(debug_dir, frame_name + "_removed_abs.png"))

    removed_gray = removed_abs.mean(dim=1, keepdim=True)
    removed_vis = torch.clamp(removed_gray * float(gain), 0.0, 1.0)
    save_image_tensor(removed_vis, os.path.join(debug_dir, frame_name + "_removed_vis.png"))

    removed_mask = (removed_gray > 0.01).float()
    save_image_tensor(removed_mask, os.path.join(debug_dir, frame_name + "_removed_mask.png"))


def infer_task(
    net,
    dataset,
    task,
    output_root,
    split_name,
    device,
    prior_strength=1.0,
    save_prior_maps=False,
    save_prefiltered_inputs=False,
    prior_bypass_model=False,
    save_filter_debug=False,
    filter_debug_gain=8.0,
):
    output_dir = os.path.join(output_root, task, split_name)
    os.makedirs(output_dir, exist_ok=True)
    prior_dir = os.path.join(output_root, "prior_maps", task, split_name)
    prefilter_dir = os.path.join(output_root, "prefiltered_inputs", task, split_name)
    filter_debug_dir = os.path.join(output_root, "filter_debug", task, split_name)

    if len(dataset) == 0:
        raise RuntimeError(f"No input images found for task={task}")
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(testloader):
            if len(batch) == 3:
                ([degraded_name], degrad_patch, prior_patch) = batch
                original_patch = None
                if save_prior_maps:
                    _save_prior_visuals(prior_patch, prior_dir, degraded_name[0])
                if save_prefiltered_inputs:
                    os.makedirs(prefilter_dir, exist_ok=True)
                    save_image_tensor(degrad_patch, os.path.join(prefilter_dir, degraded_name[0] + ".png"))
                prior_patch = prior_patch.to(device)
            elif len(batch) == 4:
                ([degraded_name], degrad_patch, prior_patch, original_patch) = batch
                if save_prior_maps:
                    _save_prior_visuals(prior_patch, prior_dir, degraded_name[0])
                if save_prefiltered_inputs:
                    os.makedirs(prefilter_dir, exist_ok=True)
                    save_image_tensor(degrad_patch, os.path.join(prefilter_dir, degraded_name[0] + ".png"))
                if save_filter_debug:
                    _save_filter_debug_visuals(
                        original_patch,
                        degrad_patch,
                        filter_debug_dir,
                        degraded_name[0],
                        gain=filter_debug_gain,
                    )
                prior_patch = prior_patch.to(device)
            else:
                ([degraded_name], degrad_patch) = batch
                prior_patch = None

            degrad_patch = degrad_patch.to(device)
            if prior_bypass_model and prior_patch is not None:
                restored = degrad_patch
            else:
                restored = net(degrad_patch, prior_patch, prior_strength=prior_strength)
            save_image_tensor(restored, os.path.join(output_dir, degraded_name[0] + ".png"))


def _resolve_ckpt_path(ckpt_name):
    if os.path.isfile(ckpt_name):
        return ckpt_name
    return os.path.join("ckpt", ckpt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument(
        '--mode',
        type=int,
        default=5,
        help=(
            "0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, "
            "4 for enhance, 5 for all four tasks (derain/dehaze/deblur/enhance), "
            "6 for all five tasks"
        )
    )

    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='path of test deblur images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='path of test enhancement images')
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='path of test denoise images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='path of test derain images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='path of test dehaze images')

    parser.add_argument('--output_path', type=str, default="AdaIR_results/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="ckpt/adair5d.ckpt", help='checkpoint name or full path')
    parser.add_argument(
        '--use_video_prior',
        action='store_true',
        help='Enable temporal prior extraction and inject it into AdaIR frequency modules.',
    )
    parser.add_argument(
        '--prior_window_size',
        type=int,
        default=9,
        help='Temporal window size used to build prior map (odd value recommended).',
    )
    parser.add_argument(
        '--prior_align_mode',
        type=str,
        default='none',
        choices=['none', 'ecc'],
        help='Temporal alignment before prior extraction.',
    )
    parser.add_argument(
        '--prior_warp_mode',
        type=str,
        default='affine',
        choices=['translation', 'euclidean', 'affine', 'homography'],
        help='ECC warp mode when --prior_align_mode ecc is used.',
    )
    parser.add_argument(
        '--prior_temporal_semantics',
        type=str,
        default='stable',
        choices=['transient', 'stable'],
        help=(
            "Temporal prior semantics: transient assumes rain/snow are short-lived; "
            "stable assumes rain/snow are in stable temporal bands."
        ),
    )
    parser.add_argument(
        '--prior_stable_band_ratio',
        type=float,
        default=0.10,
        help='Low-frequency band ratio for stable semantics in temporal FFT, range (0, 0.5).',
    )
    parser.add_argument(
        '--prior_prefilter_alpha',
        type=float,
        default=0.0,
        help='Deterministic prior-guided prefilter blend strength before AdaIR, range [0, 1].',
    )
    parser.add_argument(
        '--prior_prefilter_threshold',
        type=float,
        default=0.35,
        help='Artifact threshold used by deterministic prior prefilter, range [0, 0.95].',
    )
    parser.add_argument(
        '--prior_prefilter_mask_cap',
        type=float,
        default=0.65,
        help='Upper bound of prefilter blend mask to avoid over-smoothing, range [0, 1].',
    )
    parser.add_argument(
        '--prior_detail_protect',
        type=float,
        default=0.75,
        help='Edge/detail protection strength during prefiltering, range [0, 1].',
    )
    parser.add_argument(
        '--prior_artifact_relax',
        type=float,
        default=0.8,
        help='Relax edge protection in high-artifact areas, range [0, 2].',
    )
    parser.add_argument(
        '--prior_residual_boost',
        type=float,
        default=0.55,
        help='Boost filtering where current frame differs from temporal background, range [0, 2].',
    )
    parser.add_argument(
        '--save_prefiltered_inputs',
        action='store_true',
        help='Save prior-prefiltered input frames for debugging.',
    )
    parser.add_argument(
        '--save_filter_debug',
        action='store_true',
        help='Save original/filtered/removed visualizations of prior prefiltering.',
    )
    parser.add_argument(
        '--filter_debug_gain',
        type=float,
        default=8.0,
        help='Gain factor for removed-content heatmap visualization.',
    )
    parser.add_argument(
        '--prior_bypass_model',
        action='store_true',
        help='Skip AdaIR and output prefiltered inputs directly when prior is available.',
    )
    parser.add_argument(
        '--prior_strength',
        type=float,
        default=0.7,
        help='Strength multiplier for temporal-prior injection in AdaIR.',
    )
    parser.add_argument(
        '--save_prior_maps',
        action='store_true',
        help='Save prior maps and per-channel maps for each frame.',
    )
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    ckpt_path = _resolve_ckpt_path(args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    dehaze_splits = ["SOTS/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    derain_paths = _resolve_task_paths(args.derain_path, derain_splits)
    dehaze_paths = _resolve_task_paths(args.dehaze_path, dehaze_splits)
    deblur_paths = _resolve_task_paths(args.gopro_path, deblur_splits)
    enhance_paths = _resolve_task_paths(args.enhance_path, enhance_splits)

    denoise_tests = []
    if args.mode in [0, 6]:
        base_path = args.denoise_path
        for split in denoise_splits:
            args.denoise_path = os.path.join(base_path, split)
            denoise_tests.append((split.strip('/'), DenoiseTestDataset(args)))

    print(f"Using checkpoint: {ckpt_path}")
    if args.use_video_prior:
        print(
            (
                "Video prior enabled: window_size={}, align_mode={}, warp_mode={}, "
                "semantics={}, stable_band_ratio={}, prefilter_alpha={}, "
                "prefilter_threshold={}, prefilter_mask_cap={}, detail_protect={}, "
                "artifact_relax={}, residual_boost={}, "
                "strength={}, save_prior_maps={}, "
                "save_prefiltered_inputs={}, save_filter_debug={}, "
                "filter_debug_gain={}, prior_bypass_model={}"
            ).format(
                args.prior_window_size,
                args.prior_align_mode,
                args.prior_warp_mode,
                args.prior_temporal_semantics,
                args.prior_stable_band_ratio,
                args.prior_prefilter_alpha,
                args.prior_prefilter_threshold,
                args.prior_prefilter_mask_cap,
                args.prior_detail_protect,
                args.prior_artifact_relax,
                args.prior_residual_boost,
                args.prior_strength,
                args.save_prior_maps,
                args.save_prefiltered_inputs,
                args.save_filter_debug,
                args.filter_debug_gain,
                args.prior_bypass_model,
            )
        )
    net = AdaIRModel(prior_strength=args.prior_strength)
    net = _load_lightning_weights(net, ckpt_path, map_location=device).to(device)
    net.eval()

    if args.mode == 0:
        for split_name, testset in denoise_tests:
            for sigma in [15, 25, 50]:
                print(f"Start denoise split={split_name} sigma={sigma} ...")
                infer_denoise(net, testset, sigma, args.output_path, split_name, device)

    elif args.mode == 1:
        if not derain_paths:
            raise FileNotFoundError(f"No derain test set found under {args.derain_path}")
        for path in derain_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start derain split={split_name} ...")
            args.derain_path = path
            derain_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                derain_set,
                "derain",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    elif args.mode == 2:
        if not dehaze_paths:
            raise FileNotFoundError(f"No dehaze test set found under {args.dehaze_path}")
        for path in dehaze_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start dehaze split={split_name} ...")
            args.dehaze_path = path
            dehaze_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                dehaze_set,
                "dehaze",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    elif args.mode == 3:
        if not deblur_paths:
            raise FileNotFoundError(f"No deblur test set found under {args.gopro_path}")
        for path in deblur_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start deblur split={split_name} ...")
            args.gopro_path = path
            deblur_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                deblur_set,
                "deblur",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    elif args.mode == 4:
        if not enhance_paths:
            raise FileNotFoundError(f"No enhance test set found under {args.enhance_path}")
        for path in enhance_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start enhance split={split_name} ...")
            args.enhance_path = path
            enhance_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                enhance_set,
                "enhance",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    elif args.mode == 5:
        if not derain_paths:
            raise FileNotFoundError(f"No derain test set found under {args.derain_path}")
        for path in derain_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start derain split={split_name} ...")
            args.derain_path = path
            derain_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                derain_set,
                "derain",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not dehaze_paths:
            raise FileNotFoundError(f"No dehaze test set found under {args.dehaze_path}")
        for path in dehaze_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start dehaze split={split_name} ...")
            args.dehaze_path = path
            dehaze_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                dehaze_set,
                "dehaze",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not deblur_paths:
            raise FileNotFoundError(f"No deblur test set found under {args.gopro_path}")
        for path in deblur_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start deblur split={split_name} ...")
            args.gopro_path = path
            deblur_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                deblur_set,
                "deblur",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not enhance_paths:
            raise FileNotFoundError(f"No enhance test set found under {args.enhance_path}")
        for path in enhance_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start enhance split={split_name} ...")
            args.enhance_path = path
            enhance_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                enhance_set,
                "enhance",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    elif args.mode == 6:
        for split_name, testset in denoise_tests:
            for sigma in [15, 25, 50]:
                print(f"Start denoise split={split_name} sigma={sigma} ...")
                infer_denoise(net, testset, sigma, args.output_path, split_name, device)

        if not derain_paths:
            raise FileNotFoundError(f"No derain test set found under {args.derain_path}")
        for path in derain_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start derain split={split_name} ...")
            args.derain_path = path
            derain_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                derain_set,
                "derain",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not dehaze_paths:
            raise FileNotFoundError(f"No dehaze test set found under {args.dehaze_path}")
        for path in dehaze_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start dehaze split={split_name} ...")
            args.dehaze_path = path
            dehaze_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                dehaze_set,
                "dehaze",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not deblur_paths:
            raise FileNotFoundError(f"No deblur test set found under {args.gopro_path}")
        for path in deblur_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start deblur split={split_name} ...")
            args.gopro_path = path
            deblur_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                deblur_set,
                "deblur",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

        if not enhance_paths:
            raise FileNotFoundError(f"No enhance test set found under {args.enhance_path}")
        for path in enhance_paths:
            split_name = os.path.basename(os.path.normpath(path))
            print(f"Start enhance split={split_name} ...")
            args.enhance_path = path
            enhance_set = _build_restore_dataset(_resolve_input_dir(path), args)
            infer_task(
                net,
                enhance_set,
                "enhance",
                args.output_path,
                split_name,
                device,
                prior_strength=args.prior_strength,
                save_prior_maps=args.save_prior_maps,
                save_prefiltered_inputs=args.save_prefiltered_inputs,
                prior_bypass_model=args.prior_bypass_model,
                save_filter_debug=args.save_filter_debug,
                filter_debug_gain=args.filter_debug_gain,
            )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
