#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from datetime import datetime

import cv2
import numpy as np
import torch


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def natural_key(text):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def collect_images(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    mapping = {}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in VALID_EXTS:
            continue
        # Keep first occurrence if duplicate stem exists.
        mapping.setdefault(stem, path)
    return mapping


def read_rgb_float01(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def crop_to_min_shape(images):
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    return [im[:h, :w] for im in images]


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _ssim_single_channel(pred_ch, gt_ch):
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu1 = cv2.GaussianBlur(pred_ch, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gt_ch, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(pred_ch * pred_ch, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gt_ch * gt_ch, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(pred_ch * gt_ch, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_ssim(pred, gt):
    if pred.ndim == 2:
        return _ssim_single_channel(pred, gt)
    return float(np.mean([_ssim_single_channel(pred[..., c], gt[..., c]) for c in range(pred.shape[2])]))


class LPIPSMetric:
    def __init__(self, device, net="alex"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires package `lpips`. Install with: pip install lpips"
            ) from exc
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, pred, gt):
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred_t = pred_t * 2.0 - 1.0
        gt_t = gt_t * 2.0 - 1.0
        with torch.no_grad():
            return float(self.model(pred_t, gt_t).mean().item())


def compute_flow_farneback(prev_ref, curr_ref):
    prev_gray = cv2.cvtColor((prev_ref * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr_ref * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def warp_image_with_flow(prev_img, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(
        prev_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    valid = (map_x >= 0) & (map_x <= (w - 1)) & (map_y >= 0) & (map_y <= (h - 1))
    return warped, valid


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR / SSIM / LPIPS / Warp_Error from image folders."
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground-truth folder")
    parser.add_argument("--pred_dir", type=str, required=True, help="Prediction folder")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input folder (required if --warp_ref input)",
    )
    parser.add_argument(
        "--warp_ref",
        type=str,
        default="gt",
        choices=["gt", "input", "pred"],
        help="Reference sequence for optical flow in Warp_Error.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for LPIPS",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output prefix. Default: metrics_<pred_dir_name>_<timestamp>",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gt_map = collect_images(args.gt_dir)
    pred_map = collect_images(args.pred_dir)
    input_map = collect_images(args.input_dir) if args.input_dir else {}

    common_stems = sorted(set(gt_map.keys()) & set(pred_map.keys()), key=natural_key)
    if not common_stems:
        raise RuntimeError("No matched frames between gt_dir and pred_dir")

    if args.warp_ref == "input" and not args.input_dir:
        raise ValueError("--warp_ref input requires --input_dir")

    device = torch.device(args.device)
    lpips_metric = LPIPSMetric(device=device, net=args.lpips_net)

    frame_rows = []
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    for stem in common_stems:
        gt = read_rgb_float01(gt_map[stem])
        pred = read_rgb_float01(pred_map[stem])
        pred, gt = crop_to_min_shape([pred, gt])

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        lpips = lpips_metric(pred, gt)

        frame_rows.append(
            {
                "frame": stem,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
            }
        )
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lpips_vals.append(lpips)

    warp_rows = []
    warp_vals = []
    for idx in range(1, len(common_stems)):
        prev_stem = common_stems[idx - 1]
        curr_stem = common_stems[idx]

        if args.warp_ref == "gt":
            prev_ref = read_rgb_float01(gt_map[prev_stem])
            curr_ref = read_rgb_float01(gt_map[curr_stem])
        elif args.warp_ref == "pred":
            prev_ref = read_rgb_float01(pred_map[prev_stem])
            curr_ref = read_rgb_float01(pred_map[curr_stem])
        else:
            if prev_stem not in input_map or curr_stem not in input_map:
                continue
            prev_ref = read_rgb_float01(input_map[prev_stem])
            curr_ref = read_rgb_float01(input_map[curr_stem])

        prev_pred = read_rgb_float01(pred_map[prev_stem])
        curr_pred = read_rgb_float01(pred_map[curr_stem])

        prev_ref, curr_ref, prev_pred, curr_pred = crop_to_min_shape(
            [prev_ref, curr_ref, prev_pred, curr_pred]
        )

        flow = compute_flow_farneback(prev_ref, curr_ref)
        warped_prev_pred, valid = warp_image_with_flow(prev_pred, flow)
        if np.any(valid):
            err = np.abs(warped_prev_pred - curr_pred)
            warp_err = float(np.mean(err[valid]))
            warp_vals.append(warp_err)
            warp_rows.append(
                {
                    "pair": f"{prev_stem}->{curr_stem}",
                    "warp_error": warp_err,
                }
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        pred_name = os.path.basename(os.path.normpath(args.pred_dir)) or "pred"
        out_prefix = f"metrics_{pred_name}_{stamp}"

    csv_path = f"{out_prefix}.csv"
    warp_csv_path = f"{out_prefix}_warp.csv"
    json_path = f"{out_prefix}_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(row)

    with open(warp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "warp_error"])
        writer.writeheader()
        for row in warp_rows:
            writer.writerow(row)

    summary = {
        "n_frames": len(frame_rows),
        "n_pairs_warp": len(warp_rows),
        "warp_ref": args.warp_ref,
        "metrics": {
            "psnr": summarize(psnr_vals),
            "ssim": summarize(ssim_vals),
            "lpips": summarize(lpips_vals),
            "warp_error": summarize(warp_vals),
        },
        "paths": {
            "gt_dir": args.gt_dir,
            "pred_dir": args.pred_dir,
            "input_dir": args.input_dir,
            "csv": csv_path,
            "warp_csv": warp_csv_path,
            "summary_json": json_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# 常用：用 input 序列估计光流
# python eval_metrics.py \
#   --gt_dir test/gt \
#   --pred_dir test/rainy \
#   --input_dir test/input \
#   --warp_ref input \
#   --out_prefix test/metrics_eval_rainy


# 常用：用 input 序列估计光流
## python eval_metrics.py  --gt_dir test/gt --pred_dir test/rainy  --input_dir test/input   --warp_ref input   --out_prefix test/metrics_eval_rainy

## python eval_metrics.py  --gt_dir test/gt --pred_dir AdaIR_results_train/derain/6  --input_dir data/test/derain/6/input   --warp_ref input   --out_prefix test/metrics_eval_rainy
