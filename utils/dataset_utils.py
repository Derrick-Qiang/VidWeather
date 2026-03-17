import os
import random
import copy
import re
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
from utils.video_prior_utils import TemporalPriorExtractor

try:
    import cv2
except ImportError:
    cv2 = None

    
class AdaIRTrainDataset(Dataset):
    def __init__(self, args):
        super(AdaIRTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6}
        self.id_to_task = {v: k for k, v in self.de_dict.items()}
        self.use_video_prior_train = bool(getattr(self.args, "use_video_prior_train", False))
        self.prior_min_frames = max(int(getattr(self.args, "prior_min_frames", 3)), 3)
        requested_tasks = getattr(
            self.args,
            "prior_train_tasks",
            ['derain', 'dehaze', 'deblur', 'enhance'],
        )
        self.prior_task_ids = {
            self.de_dict[name]
            for name in requested_tasks
            if name in self.de_dict
        }
        self.valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self.temporal_prior_cache = {}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):
        temp_ids = []

        image_list = os.listdir(os.path.join(self.args.gopro_dir, 'blur/'))
        temp_ids = image_list
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        repeat_factor = max(int(getattr(self.args, "deblur_repeat_factor", 5)), 1)
        self.deblur_ids = self.deblur_ids * repeat_factor
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.enhance_dir, 'low/'))
        temp_ids = image_list
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids * 20
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = os.path.join(self.args.data_file_dir, "rainy", "rainTrain.txt")
        if os.path.isfile(rs):
            with open(rs) as f:
                for id_ in f:
                    rel_path = id_.strip()
                    if not rel_path:
                        continue
                    rain_path = os.path.join(self.args.derain_dir, rel_path)
                    if os.path.isfile(rain_path):
                        temp_ids.append(rain_path)

        # Fallback for datasets that use rainy/*.jpg and gt/*.jpg naming.
        if not temp_ids:
            rainy_dir = os.path.join(self.args.derain_dir, "rainy")
            if not os.path.isdir(rainy_dir):
                raise FileNotFoundError("Derain rainy directory not found: {}".format(rainy_dir))
            temp_ids = [
                os.path.join(rainy_dir, name)
                for name in os.listdir(rainy_dir)
                if os.path.isfile(os.path.join(rainy_dir, name))
            ]

        temp_ids.sort()
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        repeat_factor = max(int(getattr(self.args, "derain_repeat_factor", 120)), 1)
        self.rs_ids = self.rs_ids * repeat_factor

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    @staticmethod
    def _natural_key(text):
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]

    @staticmethod
    def _scene_key_from_name(file_name):
        stem = os.path.splitext(os.path.basename(file_name))[0]
        if "-" not in stem:
            return None
        prefix = stem.split("-", 1)[0]
        return prefix if prefix else None

    def _is_prior_task(self, de_id):
        return self.use_video_prior_train and de_id in self.prior_task_ids

    def _build_spatial_prior(self, image_rgb):
        image = image_rgb.astype(np.float32) / 255.0
        if cv2 is not None:
            bg = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        else:
            bg = image

        return np.clip(bg, 0.0, 1.0).astype(np.float32)

    def _get_temporal_prior(self, degraded_path):
        if not degraded_path or not os.path.isfile(degraded_path):
            return None

        parent = os.path.dirname(degraded_path)
        basename = os.path.basename(degraded_path)
        scene_key = self._scene_key_from_name(basename)
        cache_key = (parent, scene_key) if scene_key is not None else parent
        cache_entry = self.temporal_prior_cache.get(cache_key, None)

        if cache_entry is None:
            frame_paths = [
                os.path.join(parent, name)
                for name in os.listdir(parent)
                if os.path.isfile(os.path.join(parent, name))
                and os.path.splitext(name)[1].lower() in self.valid_exts
            ]
            frame_paths = sorted(frame_paths, key=lambda p: self._natural_key(os.path.basename(p)))
            if scene_key is not None:
                scene_frame_paths = [
                    path for path in frame_paths
                    if self._scene_key_from_name(os.path.basename(path)) == scene_key
                ]
                if len(scene_frame_paths) >= self.prior_min_frames:
                    frame_paths = scene_frame_paths
            if len(frame_paths) < self.prior_min_frames:
                self.temporal_prior_cache[cache_key] = False
                return None

            index_map = {os.path.basename(path): i for i, path in enumerate(frame_paths)}
            try:
                extractor = TemporalPriorExtractor(
                    frame_paths=frame_paths,
                    window_size=getattr(self.args, "prior_window_size", 9),
                    align_mode=getattr(self.args, "prior_align_mode", "none"),
                    warp_mode=getattr(self.args, "prior_warp_mode", "affine"),
                    temporal_semantics=getattr(self.args, "prior_temporal_semantics", "stable"),
                    stable_band_ratio=getattr(self.args, "prior_stable_band_ratio", 0.15),
                    crop_base=16,
                )
            except Exception:
                self.temporal_prior_cache[cache_key] = False
                return None

            cache_entry = (extractor, index_map)
            self.temporal_prior_cache[cache_key] = cache_entry

        if cache_entry is False:
            return None

        extractor, index_map = cache_entry
        center_idx = index_map.get(basename, None)
        if center_idx is None:
            return None

        try:
            prior = extractor.get_prior(center_idx)
        except Exception:
            return None
        return np.clip(prior.astype(np.float32), 0.0, 1.0)

    def _get_training_prior(self, degraded_path, degraded_img, de_id):
        if not self.use_video_prior_train:
            return None

        temporal_prior = None
        if self._is_prior_task(de_id):
            temporal_prior = self._get_temporal_prior(degraded_path)

        if temporal_prior is None:
            prior = self._build_spatial_prior(degraded_img)
        else:
            prior = temporal_prior

        if prior.shape[:2] != degraded_img.shape[:2]:
            h, w = degraded_img.shape[:2]
            prior_uint8 = np.clip(prior * 255.0, 0.0, 255.0).astype(np.uint8)
            prior_uint8 = np.array(Image.fromarray(prior_uint8).resize((w, h), resample=Image.BILINEAR))
            prior = np.clip(prior_uint8.astype(np.float32) / 255.0, 0.0, 1.0)
        return prior

    def _crop_patch(self, *images):
        H = images[0].shape[0]
        W = images[0].shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patches = [
            image[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
            for image in images
        ]

        return patches if len(patches) > 1 else patches[0]

    def _get_gt_name(self, rainy_name):
        rainy_root = os.path.dirname(os.path.dirname(rainy_name))
        rainy_file = os.path.basename(rainy_name)

        # Legacy naming: rainy/rain-13.png -> gt/norain-13.png
        if rainy_file.startswith("rain-"):
            legacy_gt = os.path.join(rainy_root, "gt", "norain-" + rainy_file.split("rain-", 1)[1])
            if os.path.isfile(legacy_gt):
                return legacy_gt

        # Paired naming: rainy/1-0000.jpg -> gt/1-0000.jpg
        return os.path.join(rainy_root, "gt", rainy_file)


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        prior_patch = None
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
            prior_patch = self._get_training_prior(None, degrad_patch, de_id)
        else:
            degraded_path = None
            if de_id == 3:
                # Rain Streak Removal
                degraded_path = sample["clean_id"]
                degrad_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degraded_path = sample["clean_id"]
                degrad_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # Deblur with Gopro set
                degraded_path = os.path.join(self.args.gopro_dir, 'blur/', sample["clean_id"])
                degrad_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'sharp/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
                degraded_path = os.path.join(self.args.enhance_dir, 'low/', sample["clean_id"])
                degrad_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'gt/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_enhance_name(sample["clean_id"])

            prior_img = self._get_training_prior(degraded_path, degrad_img, de_id)
            if prior_img is not None:
                prior_uint8 = np.clip(prior_img * 255.0, 0.0, 255.0).astype(np.uint8)
                degrad_patch, clean_patch, prior_patch = random_augmentation(
                    *self._crop_patch(degrad_img, clean_img, prior_uint8)
                )
                prior_patch = np.clip(prior_patch.astype(np.float32) / 255.0, 0.0, 1.0)
            else:
                degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        if self.use_video_prior_train:
            if prior_patch is None:
                prior_patch = torch.zeros_like(degrad_patch)
            else:
                prior_patch = torch.from_numpy(np.transpose(prior_patch, (2, 0, 1))).float()
            return [clean_name, de_id], degrad_patch, clean_patch, prior_patch

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.enhance_path + 'input/')
            self.ids += [self.args.enhance_path + 'input/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            input_name = degraded_name.split('/')[-1]

            # Prefer paired files with the same filename/suffix.
            same_name_gt = dir_name + input_name
            if os.path.exists(same_name_gt):
                gt_name = same_name_gt
            else:
                # Fallback to the original SOTS naming style.
                stem = os.path.splitext(input_name)[0].split('_')[0]
                png_gt = dir_name + stem + '.png'
                jpg_gt = dir_name + stem + '.jpg'
                if os.path.exists(png_gt):
                    gt_name = png_gt
                elif os.path.exists(jpg_gt):
                    gt_name = jpg_gt
                else:
                    gt_name = png_gt
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
