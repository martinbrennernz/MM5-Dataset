import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from config import config as C

from typing import Dict, List
###############################################################################
# helpers                                                                     #
###############################################################################

# _multi_ch_keys = tuple(C.multi_channel_keywords)

# def _is_single(path: str) -> bool:
#     return not any(k in path for k in _multi_keys)

###############################################################################
# dataset                                                                     #
###############################################################################

class MM5Dataset(data.Dataset):
    def __init__(self, setting: dict, split: str, preprocess=None, file_length=None):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = split
        self.preprocess = preprocess
        self.file_length = file_length

        # roots and formats for RGB and GT
        self.rgb_root   = setting['rgb_root']
        self.rgb_format = setting['rgb_format']
        self.gt_root    = setting['gt_root']
        self.gt_format  = setting['gt_format']
        self.transform_gt = setting['transform_gt']

        # active extra modalities
        active_mods = C.active_modalities()
        self.n_modal_active = len(active_mods)
        self.x_roots   = [C.x_root_folders[i - 1]      for i in active_mods]
        self.x_formats = [C.x_formats[i - 1]           for i in active_mods]
        self.x_single  = [C.x_is_single_channel[i - 1] for i in active_mods]

        # normalization stats per modality
        self.x_norm_params_list = []
        if hasattr(C, 'x_norm_stats') and len(C.x_norm_stats) == self.n_modal_active:
            for params in C.x_norm_stats:
                self.x_norm_params_list.append({
                    'mean': np.array(params['mean'], dtype=np.float32),
                    'std':  np.array(params['std'],  dtype=np.float32)
                })
        else:
            raise ValueError("Mismatch in C.x_norm_stats and active modalities")

        # class names and file listing
        self.class_names = setting['class_names']
        list_file = setting['train_source'] if split == 'train' else setting['eval_source']
        with open(list_file) as f:
            self.files = [l.strip() for l in f if l.strip()]
            if (
                split == "train"
                and getattr(C, "rare_class_multiply", False)
                and hasattr(C, "rare_files_ids")
                and hasattr(C, "rare_files_multiplication")
                and len(C.rare_files_ids) == len(C.rare_files_multiplication)
            ):
                new_files = []
                for fname in self.files:
                    try:
                        file_id = int(fname.split('.')[0])
                    except (ValueError, IndexError):
                        file_id = None
                    if file_id in C.rare_files_ids:
                        idx = C.rare_files_ids.index(file_id)
                        mult = C.rare_files_multiplication[idx]
                        new_files.extend([fname] * mult)
                    else:
                        new_files.append(fname)
                self.files = new_files

        if file_length:
            reps = file_length // len(self.files)
            rem  = file_length % len(self.files)
            idxs = torch.randperm(len(self.files))[:rem].tolist()
            self.rep_files = self.files * reps + [self.files[i] for i in idxs]
        else:
            self.rep_files = self.files

    def __len__(self):
        return len(self.rep_files)

    def get_length(self):
        # For compatibility with evaluators expecting get_length()
        return len(self)

    @staticmethod
    def _read_img(path: str, single: bool) -> np.ndarray:
        if single:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = img[:, :, None]
        else:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if img is None:
            raise FileNotFoundError(path)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        name = self.rep_files[idx]
        # load raw RGB
        rgb_path = osp.join(self.rgb_root, name + self.rgb_format)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # load label
        label = None
        if self.split != 'test':
            gt_path = osp.join(self.gt_root, name + self.gt_format)
            label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if label is not None and label.ndim == 3:
                label = label[:, :, 0]
            if label is not None and self.transform_gt:
                label = label.astype(np.int32) - 1

        # load extra modalities
        modals = []
        for root, fmt in zip(self.x_roots, self.x_formats):
            mpath = osp.join(root, name + fmt)
            m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(mpath)
            if m.ndim == 2:
                m = m[:, :, None]
            elif m.shape[2] == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            modals.append(m.astype(np.float32))

        # conditionally resize to config size
        target_h, target_w = C.image_height, C.image_width
        h0, w0 = rgb.shape[:2]
        if (h0, w0) != (target_h, target_w):
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            for i, m in enumerate(modals):
                mh, mw = m.shape[:2]
                if (mh, mw) != (target_h, target_w):
                    interp = cv2.INTER_NEAREST if m.ndim == 2 or m.shape[2] == 1 else cv2.INTER_LINEAR
                    m_res = cv2.resize(m, (target_w, target_h), interpolation=interp)
                    if m_res.ndim == 2:
                        m_res = m_res[:, :, None]
                    modals[i] = m_res

        # preprocess (normalize, augment, etc.)
        rgb_new, label_new, modal_new = self.preprocess(
            rgb, label, modals, self.x_norm_params_list)

        if modal_new.ndim == 2:
            modal_new = modal_new[:, :, None]
    
        # to torch
        sample = {
            'rgb':     torch.from_numpy(rgb_new.transpose(2,0,1)).float(),
            'modal_x': torch.from_numpy(modal_new.transpose(2,0,1)).float(),
            'fn':      name,
            'n':       len(self.files)
        }
        sample['data'] = sample['rgb']
        sample['label'] = torch.from_numpy(label_new).long() if label_new is not None else None
        return sample

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            return ''.join(str((n>>y)&1) for y in range(count-1, -1, -1))
        N=41; cmap=np.zeros((N,3),dtype=np.uint8)
        for i in range(N):
            r=g=b=0; tmp=i
            for j in range(7):
                bits=uint82bin(tmp)
                r^=(int(bits[-1])<<(7-j));
                g^=(int(bits[-2])<<(7-j));
                b^=(int(bits[-3])<<(7-j));
                tmp>>=3
            cmap[i]=[r,g,b]
        return cmap.tolist()

# collate

def multimodal_collate(batch):
    keys=['data','label','modal_x']
    coll={k:[s[k] for s in batch if k in s] for k in keys}
    for k in coll: coll[k]=torch.stack(coll[k],0)
    return coll
