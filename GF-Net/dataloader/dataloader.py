from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .MM5Dataset import MM5Dataset, multimodal_collate   # ← from MM5Dataset.py
from config import config as C
from utils.transforms import (
    normalize,
    generate_random_crop_pos,
    random_crop_pad_to_shape,
)
import random
import cv2
import numpy as np

# from typing import Dict, List, Optional

def get_train_loader(engine):
    """
    Returns
    -------
    train_loader  : DataLoader
    train_sampler : DistributedSampler | None
    """

    data_setting = {
        'rgb_root': C.rgb_root_folder,
        'rgb_format': C.rgb_format,
        'gt_root': C.gt_root_folder,
        'gt_format': C.gt_format,
        'transform_gt': C.gt_transform,
        'x_roots': C.x_root_folders,
        'x_formats': C.x_formats,
        'x_single_channels': C.x_is_single_channel,
        'class_names': C.class_names,
        'train_source': C.train_source,
        'eval_source': C.eval_source
    }
        
    # preprocess object from dataset.py
    train_preprocess = TrainPre()

    # train_dataset = MM5Dataset(data_setting, "train", train_preprocess, C.batch_size * C.niters_per_epoch)

    # MM5Dataset handles X modalities internally
    file_len = C.batch_size * C.niters_per_epoch
    train_dataset = MM5Dataset(data_setting, "train", train_preprocess,
                               file_length=file_len)

    # distributed settings
    train_sampler = None
    is_shuffle    = True
    batch_size    = C.batch_size

    if engine.distributed:
        train_sampler = dtrain_sampler = DistributedSampler(train_dataset)
        batch_size    = C.batch_size // engine.world_size
        is_shuffle    = False

    train_loader = DataLoader(
        train_dataset,
        batch_size   = batch_size,
        num_workers  = C.num_workers,
        drop_last    = True,
        shuffle      = is_shuffle,
        pin_memory   = True,
        sampler      = train_sampler,      # this stays the same
        collate_fn   = multimodal_collate,
    )

    return train_loader, train_sampler


def random_mirror(rgb, gt, modal_xs):
    if np.random.rand() < 0.5:
        rgb = cv2.flip(rgb, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)
        modal_xs = [cv2.flip(modal_x, 1) for modal_x in modal_xs]
    return rgb, gt, modal_xs

def random_scale(rgb, gt, modal_xs, scale_array):
    scale = np.random.choice(scale_array)
    new_h, new_w = int(rgb.shape[0] * scale), int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    modal_xs_resized = [
        cv2.resize(modal_x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        for modal_x in modal_xs
    ]
    return rgb, gt, modal_xs_resized, scale

def random_rotate_zoom(rgb, gt, modal_xs, max_angle=20, zoom_range=(1.0, 1.2)):
    # --- Random rotation ---
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = rgb.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rgb_r = cv2.warpAffine(rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gt_r = cv2.warpAffine(gt, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) if gt is not None else None
    modal_xs_r = [cv2.warpAffine(modal_x, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                  for modal_x in modal_xs]

    # --- Random zoom ---
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    zh, zw = int(h * zoom_factor), int(w * zoom_factor)
    rgb_z = cv2.resize(rgb_r, (zw, zh), interpolation=cv2.INTER_LINEAR)
    gt_z = cv2.resize(gt_r, (zw, zh), interpolation=cv2.INTER_NEAREST) if gt_r is not None else None
    modal_xs_z = [
        cv2.resize(modal_x, (zw, zh), interpolation=cv2.INTER_LINEAR)
        for modal_x in modal_xs_r
    ]

    # --- Center crop back to original size ---
    start_x = (zw - w) // 2
    start_y = (zh - h) // 2
    rgb_crop = rgb_z[start_y:start_y+h, start_x:start_x+w]
    gt_crop = gt_z[start_y:start_y+h, start_x:start_x+w] if gt_z is not None else None
    modal_xs_crop = [modal_x[start_y:start_y+h, start_x:start_x+w] for modal_x in modal_xs_z]

    return rgb_crop, gt_crop, modal_xs_crop

class TrainPre:
    """
    TrainPre supports multi-channel and multi-modal input.
    For rare class images, applies one randomly chosen operation (mirror, scale, rotate/zoom).
    For other images, applies standard mirror and scale.
    """
    def __init__(self):
        self.rgb_norm_mean = C.norm_mean
        self.rgb_norm_std  = C.norm_std
        self.crop_size = (C.image_height, C.image_width)
        self.rare_class_ids = getattr(C, 'rare_class_ids', [])

    def __call__(self, rgb, gt, modal_xs, modal_xs_norm_params):
        is_rare = False
        if gt is not None and len(self.rare_class_ids) > 0:
            unique_classes = np.unique(gt)
            is_rare = any(cls in self.rare_class_ids for cls in unique_classes)
        
        if is_rare:
            # Pick one augmentation randomly for rare classes
            aug_ops = []
            aug_ops.append(lambda r, g, m: random_mirror(r, g, m))
            if C.train_scale_array is not None:
                aug_ops.append(lambda r, g, m: random_scale(r, g, m, C.train_scale_array)[:3])
            aug_ops.append(lambda r, g, m: random_rotate_zoom(r, g, m, max_angle=20, zoom_range=(1.0, 1.2)))
            op = np.random.choice(aug_ops)
            rgb, gt, modal_xs = op(rgb, gt, modal_xs)
        else:
            # Standard augmentations
            rgb, gt, modal_xs = random_mirror(rgb, gt, modal_xs)
            if C.train_scale_array is not None:
                rgb, gt, modal_xs, _ = random_scale(rgb, gt, modal_xs, C.train_scale_array)

        # Normalize RGB
        rgb = normalize(rgb, self.rgb_norm_mean, self.rgb_norm_std)

        # Normalize each modality separately using provided norm params
        normalized_modal_xs = []
        for mod_x, norm_params in zip(modal_xs, modal_xs_norm_params):
            if mod_x.ndim == 2:
                mod_x = mod_x[:, :, None]
            c = mod_x.shape[2]
            mean, std = norm_params['mean'], norm_params['std']
            if len(mean) != c:
                mean = np.repeat(mean[0], c)
                std  = np.repeat(std[0], c)
            normalized_mod_x = normalize(mod_x, mean, std)
            normalized_modal_xs.append(normalized_mod_x)
        modal_x_combined = np.concatenate(normalized_modal_xs, axis=2)

        # Crop/pad
        crop_pos = generate_random_crop_pos(rgb.shape[:2], self.crop_size)
        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, self.crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x_combined, crop_pos, self.crop_size, 0)
        p_gt = None
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, self.crop_size, 255)

        return p_rgb, p_gt, p_modal_x

class TrainPreOld:
    """
    TrainPre supports multi-channel and multi-modal input.
    Operations:
      1) random mirror
      2) random scale
      3) normalize (per modality)
      4) random crop/pad
    """
    def __init__(self):
        self.rgb_norm_mean = C.norm_mean
        self.rgb_norm_std  = C.norm_std
        self.crop_size = (C.image_height, C.image_width)

    def __call__(self, rgb, gt, modal_xs, modal_xs_norm_params):
        # 1) random mirror
        rgb, gt, modal_xs = random_mirror(rgb, gt, modal_xs)

        # 2) random scale
        if C.train_scale_array is not None:
            rgb, gt, modal_xs, _ = random_scale(rgb, gt, modal_xs, C.train_scale_array)

        # 3) normalize RGB
        rgb = normalize(rgb, self.rgb_norm_mean, self.rgb_norm_std)

        # Normalize each modality separately using provided norm params
        normalized_modal_xs = []
        for mod_x, norm_params in zip(modal_xs, modal_xs_norm_params):
            if mod_x.ndim == 2:
                mod_x = mod_x[:, :, None]  # Ensure channel dimension

            c = mod_x.shape[2]
            mean, std = norm_params['mean'], norm_params['std']

            # Handle mismatch in provided stats (repeat single channel stats if needed)
            if len(mean) != c:
                mean = np.repeat(mean[0], c)
                std  = np.repeat(std[0], c)

            normalized_mod_x = normalize(mod_x, mean, std)
            normalized_modal_xs.append(normalized_mod_x)

        # Concatenate modalities along channel dimension
        modal_x_combined = np.concatenate(normalized_modal_xs, axis=2)

        # 4) random crop/pad
        crop_pos = generate_random_crop_pos(rgb.shape[:2], self.crop_size)
        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, self.crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x_combined, crop_pos, self.crop_size, 0)

        p_gt = None
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, self.crop_size, 255)

        # Return in (H, W, C) shape, dataset handles transpose later
        return p_rgb, p_gt, p_modal_x

    
class ValPreNoOp:
    def __call__(self, rgb, gt, modal_xs, modal_xs_norm_params):
        """
        A no-op validation preprocessor that just concatenates the raw
        modal_xs list back into the single combined array that the
        rest of the pipeline expects.
        """
        # modal_xs is a list of H×W×Ci arrays; stack channel-wise
        modal_xs_combined = np.concatenate(modal_xs, axis=2)
        return rgb, gt, modal_xs_combined
    