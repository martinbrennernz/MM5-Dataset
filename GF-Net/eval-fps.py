import os
import cv2
import sys
import importlib
import argparse

# Set up a minimal parser to grab only --config (before anything else)
mini_parser = argparse.ArgumentParser(add_help=False)
mini_parser.add_argument('--config', '-f', type=str, default='config',
                         help='Config file to use (without .py, must be in ./configs/). E.g., "config_small"')
mini_args, remaining_argv = mini_parser.parse_known_args()

# Dynamic config import (before using config anywhere)
config_module_name = f'configs.{mini_args.config}'
config_module_path = os.path.join(os.path.dirname(__file__), 'configs', f'{mini_args.config}.py')
if not os.path.isfile(config_module_path):
    raise FileNotFoundError(f"Config file not found: {config_module_path}")

sys.path.insert(0, os.path.dirname(__file__))
cfg_mod = importlib.import_module(config_module_name)
config = cfg_mod.config

# Patch sys.modules so all subsequent `import config` grabs this config
import types
fake_config_mod = types.ModuleType('config')
fake_config_mod.config = config
sys.modules['config'] = fake_config_mod

import numpy as np

import torch
import torch.nn as nn
import time

# from config import config as C
from utils.pyt_utils import ensure_dir, parse_devices
from utils.visualize import print_iou, show_img, get_class_colors
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.MM5Dataset import MM5Dataset, multimodal_collate
from dataloader.dataloader import ValPreNoOp as ValPre 
from models.builder import EncoderDecoder as segmodel
from PIL import Image

logger = get_logger()

# Patch sys.modules so all subsequent `import config` grabs this config
import types
fake_config_mod = types.ModuleType('config')
fake_config_mod.config = config
sys.modules['config'] = fake_config_mod

def colorize_label(label, palette):
    result_img = Image.fromarray(label.astype(np.uint8), mode='P')
    result_img.putpalette(palette)
    return result_img


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_xs = data['modal_x']
        name = data['fn']

        # Compute prediction as usual
        pred = self.sliding_eval_rgbX(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn_base = os.path.join(self.save_path, name)
        
            # -- Palette for color maps --
            class_colors = get_class_colors(config.num_classes)
            palette_list = list(np.array(class_colors).flat) + [0] * (768 - len(class_colors)*3)
        
            # 1. Save GT as color image
            if isinstance(label, torch.Tensor):
                label_np = label.cpu().numpy()
            else:
                label_np = np.asarray(label)
            colorize_label(label_np, palette_list).save(fn_base + '_gt_color.png')
        
            # 2. Save prediction as raw indices (grayscale)
            cv2.imwrite(fn_base + '_pred.png', pred.astype(np.uint8))
        
            # 3. Save prediction as color image
            colorize_label(pred.astype(np.uint8), palette_list).save(fn_base + '_pred_color.png')
        
            #logger.info(f'Saved colored GT and prediction for {name}')
        

        if self.show_image:
            colors = get_class_colors(config.num_classes)
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, img, clean, label, pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict


    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct, labeled, count = 0, 0, 0

        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--config', '-f', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    device = torch.device(f"cuda:{all_dev[0]}" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.eval()
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'x_roots': config.x_root_folders,
        'x_formats': config.x_formats,
        'x_single_channels': config.x_is_single_channel,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'eval_source': config.eval_source
    }

    total_params, trainable_params = count_parameters(network)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    val_pre = ValPre()
    dataset = MM5Dataset(data_setting, 'val', preprocess=val_pre)

    sample = dataset[0]  # Just get item 0
    # print("Single sample shapes:", sample["rgb"].shape, sample["label"].shape, ...)
    # # Optionally show it in a cv2 window here to confirm how it looks
    # rgb_u8 = sample["rgb"].astype(np.uint8)
    # window_title = f"Raw RGB: "
    # cv2.imshow(window_title, rgb_u8)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(1)

    dataset = MM5Dataset(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                  config.norm_std, network,
                                  config.eval_scale_array, config.eval_flip,
                                  all_dev, args.verbose, args.save_path,
                                  args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
    

    total_imgs = len(dataset)
    inf_times = []
    
    with torch.no_grad():
        for idx in range(total_imgs):
            sample = dataset[idx]
            img_tensor   = sample['data'].unsqueeze(0).to(device)
            modal_tensor = sample['modal_x'].unsqueeze(0).to(device)
    
            # ensure any previous kernels are done
            torch.cuda.synchronize(device)
    
            t0 = time.perf_counter()
            # forward
            out = network.encode_decode(img_tensor, modal_tensor)
            # wait for kernels again
            torch.cuda.synchronize(device)
    
            t1 = time.perf_counter()
            inf_times.append(t1 - t0)
    
    # now compute
    total_time = sum(inf_times)
    mean_time = total_time / total_imgs
    fps_overall = total_imgs / total_time
    fps_per_image = 1.0 / mean_time
    
    print(f"Validation inference time: {total_time:.3f}s over {total_imgs} images")
    print(f"→ Overall FPS: {fps_overall:.2f} images/sec")
    print(f"→ Mean latency: {mean_time*1000:.1f} ms/image  (≈ {fps_per_image:.2f} FPS)")

    try:
        from thop import profile
        rgb_shape = (1, 3, config.image_height, config.image_width)
        # Infer the number of channels for each modality
        modal_shapes = []
        for is_single in config.x_is_single_channel:
            c = 1 if is_single else 3
            modal_shapes.append((1, c, config.image_height, config.image_width))
        rgb_dummy = torch.randn(*rgb_shape).to(device)
        modal_dummy = [torch.randn(*s).to(device) for s in modal_shapes]
        # If your model expects a list/tuple of modalities, pass as [modal_dummy] or modal_dummy
        flops, params = profile(network, inputs=(rgb_dummy, modal_dummy), verbose=False)
        print(f"Measured FLOPs: {flops/1e9:.2f} GFLOPs  (batch=1, input size {config.image_height}x{config.image_width})")
    except Exception as e:
        print("FLOPs measurement failed:", e)
