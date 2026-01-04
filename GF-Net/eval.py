import os
import cv2
import argparse
import numpy as np
import sys
import importlib

import torch
import torch.nn as nn

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
from utils.pyt_utils import ensure_dir, load_model, parse_devices
from utils.visualize import print_iou, show_img, get_class_colors
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.MM5Dataset import MM5Dataset, multimodal_collate
from dataloader.dataloader import ValPreNoOp as ValPre 
from models.builder import EncoderDecoder as segmodel
from PIL import Image

logger = get_logger()

# Patch sys.modules again to ensure all modules use the same config
import types
fake_config_mod = types.ModuleType('config')
fake_config_mod.config = config
sys.modules['config'] = fake_config_mod


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, probe_only=False):
        img = data['data']
        label = data['label']
        modal_xs = data['modal_x']
        name = data['fn']
        
        # Extract multi-head labels if available
        label_thermal = data.get('label_thermal', None)
        label_uv = data.get('label_uv', None)
        
        # If probe_only mode, just do a quick model call for debug logging
        if probe_only:
            if getattr(config, 'use_multi_head_fusion', False):
                # Multi-head fusion evaluation (probe only)
                self.sliding_eval_rgbX_multihead_fusion(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device, probe_only=True)
            elif getattr(config, 'use_multi_head_decoder', False):
                # Multi-head evaluation (probe only)
                self.sliding_eval_rgbX_multihead(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device, probe_only=True)
            else:
                # Single-head evaluation (probe only)
                self.sliding_eval_rgbX(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device, probe_only=True)
            return None  # Early return for probe mode

        # Check if multi-head fusion mode is enabled
        if getattr(config, 'use_multi_head_fusion', False):
            # Multi-head fusion evaluation
            predictions = self.sliding_eval_rgbX_multihead_fusion(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device)
            results_dict = self.compute_multihead_fusion_metrics(predictions, label, label_thermal, label_uv)

            # Save multi-head fusion predictions
            if self.save_path is not None:
                self.save_multihead_fusion_predictions(predictions, name)

        # Check if multi-head mode is enabled
        elif getattr(config, 'use_multi_head_decoder', False):
            # Multi-head evaluation
            predictions = self.sliding_eval_rgbX_multihead(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device)
            results_dict = self.compute_multihead_metrics(predictions, label, label_thermal, label_uv)

            # Save multi-head predictions
            if self.save_path is not None:
                self.save_multihead_predictions(predictions, name)
        else:
            # Single-head evaluation (backward compatibility)
            pred = self.sliding_eval_rgbX(img, modal_xs, config.eval_crop_size, config.eval_stride_rate, device)
            hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
            results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

            if self.save_path is not None:
                ensure_dir(self.save_path)
                ensure_dir(f"{self.save_path}_color")

                fn = name + '.png'

                result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
                class_colors = get_class_colors(config.num_classes)
                # Convert BGR to RGB in the palette itself
                class_colors_rgb = [(b, g, r) for (r, g, b) in class_colors]
                palette_list = list(np.array(class_colors_rgb).flat)
                palette_list += [0] * (768 - len(palette_list))
                result_img.putpalette(palette_list)
                
                result_img.save(os.path.join(self.save_path + '_color', fn))

                cv2.imwrite(os.path.join(self.save_path, fn), pred)
                logger.info(f'Save the image {fn}')

        if self.show_image:
            colors = dataset.get_class_colors()
            clean = np.zeros(label.shape)
            if getattr(config, 'use_multi_head_decoder', False) and 'primary' in predictions:
                pred_to_show = predictions['primary']
            else:
                pred_to_show = pred
            comp_img = show_img(colors, config.background, img, clean, label, pred_to_show)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def sliding_eval_rgbX_multihead(self, img, modal_xs, crop_size, stride_rate, device, probe_only=False):
        """Multi-head sliding window evaluation"""
        # Convert inputs to proper format (similar to sliding_eval_rgbX)
        if isinstance(img, torch.Tensor):
            arr = img.cpu().numpy()
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            img = arr.transpose(1, 2, 0)

        if isinstance(modal_xs, torch.Tensor):
            mx = modal_xs.cpu().numpy()
            if mx.ndim == 4 and mx.shape[0] == 1:
                mx = mx[0]
            # Use only active modalities for splitting
            active_mods = config.active_modalities()
            # Map active modality indices to list positions (0-indexed)
            active_single_ch = [config.x_is_single_channel[idx] for idx in range(len(active_mods))]
            ch_counts = [1 if s else 3 for s in active_single_ch]
            splits = np.split(mx, np.cumsum(ch_counts)[:-1], axis=0)
            modal_list = [sp.transpose(1, 2, 0) for sp in splits]
        else:
            modal_list = modal_xs

        # Initialize prediction accumulators for each head
        ori_h, ori_w, _ = img.shape
        pred_accum = {
            'primary': np.zeros((ori_h, ori_w, config.num_classes), dtype=np.float32),
            'thermal': np.zeros((ori_h, ori_w, getattr(config, 'num_classes_thermal', config.num_classes)), dtype=np.float32),
            'uv': np.zeros((ori_h, ori_w, getattr(config, 'num_classes_uv', config.num_classes)), dtype=np.float32)
        }

        # If probe_only mode, just do a single quick model call for debug logging
        if probe_only:
            # Use the first scale for a quick probe
            s = self.multi_scales[0] if self.multi_scales else 1.0
            new_h, new_w = int(ori_h * s), int(ori_w * s)
            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Resize modalities
            if modal_list is not None:
                modal_scale = []
                for m in modal_list:
                    interp = cv2.INTER_NEAREST if m.ndim == 2 else cv2.INTER_LINEAR
                    modal_scale.append(
                        cv2.resize(m, (new_w, new_h), interpolation=interp)
                    )
            else:
                modal_scale = None

            # Do a quick probe call to trigger debug logging
            self.scale_process_rgbX_multihead(
                img_scale, modal_scale, (ori_h, ori_w),
                crop_size, stride_rate, device, probe_only=True
            )
            return {}  # Return empty dict for probe mode

        # Multi-scale loop
        for s in self.multi_scales:
            new_h, new_w = int(ori_h * s), int(ori_w * s)
            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Resize modalities
            if modal_list is not None:
                modal_scale = []
                for m in modal_list:
                    interp = cv2.INTER_NEAREST if m.ndim == 2 else cv2.INTER_LINEAR
                    modal_scale.append(
                        cv2.resize(m, (new_w, new_h), interpolation=interp)
                    )
            else:
                modal_scale = None

            # Process this scale
            scale_preds = self.scale_process_rgbX_multihead(
                img_scale, modal_scale, (ori_h, ori_w), 
                (crop_size, crop_size) if isinstance(crop_size, int) else crop_size,
                stride_rate, device
            )
            
            # Accumulate predictions for each head
            for head_name, pred in scale_preds.items():
                pred_accum[head_name] += pred

        # Final argmax for each head
        predictions = {}
        for head_name, accum in pred_accum.items():
            predictions[head_name] = accum.argmax(axis=2)

        return predictions

    def sliding_eval_rgbX_multihead_fusion(self, img, modal_xs, crop_size, stride_rate, device, probe_only=False):
        """Multi-head fusion sliding window evaluation"""
        # Convert inputs to proper format (similar to sliding_eval_rgbX)
        if isinstance(img, torch.Tensor):
            arr = img.cpu().numpy()
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            img = arr.transpose(1, 2, 0)

        if isinstance(modal_xs, torch.Tensor):
            mx = modal_xs.cpu().numpy()
            if mx.ndim == 4 and mx.shape[0] == 1:
                mx = mx[0]
            # Use only active modalities for splitting
            active_mods = config.active_modalities()
            # Map active modality indices to list positions (0-indexed)
            active_single_ch = [config.x_is_single_channel[idx] for idx in range(len(active_mods))]
            ch_counts = [1 if s else 3 for s in active_single_ch]
            splits = np.split(mx, np.cumsum(ch_counts)[:-1], axis=0)
            modal_list = [sp.transpose(1, 2, 0) for sp in splits]
        else:
            modal_list = modal_xs

        # Initialize prediction accumulators for each head + fused
        ori_h, ori_w, _ = img.shape
        pred_accum = {
            'primary': np.zeros((ori_h, ori_w, config.num_classes), dtype=np.float32),
            'thermal': np.zeros((ori_h, ori_w, getattr(config, 'num_classes_thermal', config.num_classes)), dtype=np.float32),
            'uv': np.zeros((ori_h, ori_w, getattr(config, 'num_classes_uv', config.num_classes)), dtype=np.float32),
            'fused': np.zeros((ori_h, ori_w, config.num_classes), dtype=np.float32)  # Fused output
        }

        # If probe_only mode, just do a single quick model call for debug logging
        if probe_only:
            # Use the first scale for a quick probe
            s = self.multi_scales[0] if self.multi_scales else 1.0
            new_h, new_w = int(ori_h * s), int(ori_w * s)
            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Resize modalities
            if modal_list is not None:
                modal_scale = []
                for m in modal_list:
                    interp = cv2.INTER_NEAREST if m.ndim == 2 else cv2.INTER_LINEAR
                    modal_scale.append(
                        cv2.resize(m, (new_w, new_h), interpolation=interp)
                    )
            else:
                modal_scale = None

            # Do a quick probe call to trigger debug logging
            self.scale_process_rgbX_multihead_fusion(
                img_scale, modal_scale, (ori_h, ori_w),
                (crop_size, crop_size) if isinstance(crop_size, int) else crop_size,
                stride_rate, device, probe_only=True
            )
            return {}  # Return empty dict for probe mode

        # Multi-scale loop
        for s in self.multi_scales:
            new_h, new_w = int(ori_h * s), int(ori_w * s)
            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Resize modalities
            if modal_list is not None:
                modal_scale = []
                for m in modal_list:
                    interp = cv2.INTER_NEAREST if m.ndim == 2 else cv2.INTER_LINEAR
                    modal_scale.append(
                        cv2.resize(m, (new_w, new_h), interpolation=interp)
                    )
            else:
                modal_scale = None

            # Process this scale (uses fusion-enabled scale processor)
            scale_preds = self.scale_process_rgbX_multihead_fusion(
                img_scale, modal_scale, (ori_h, ori_w), 
                (crop_size, crop_size) if isinstance(crop_size, int) else crop_size,
                stride_rate, device
            )
            
            # Accumulate predictions for each head + fused
            for head_name, pred in scale_preds.items():
                if head_name in pred_accum:
                    pred_accum[head_name] += pred
                else:
                    # Handle unexpected head names gracefully
                    pred_accum[head_name] = pred

            # Ensure 'fused' output exists in scale predictions
            if 'fused' not in scale_preds and 'primary' in scale_preds:
                # Use primary as fallback for missing fused output
                pred_accum['fused'] += scale_preds['primary']

        # Apply confidence router if enabled (before argmax)
        if getattr(config, 'eval_conf_router_enable', False):
            pred_accum = self.apply_confidence_router(pred_accum)

        # Final argmax for each head + fused
        predictions = {}
        for head_name, accum in pred_accum.items():
            predictions[head_name] = accum.argmax(axis=2)

        # Fallback logic: ensure 'fused' output exists
        if 'fused' not in predictions:
            if 'primary' in predictions:
                # Use primary as fallback for fused
                predictions['fused'] = predictions['primary']
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Stage-wise fuser 'fused' output missing, using 'primary' as fallback")
            else:
                # If neither fused nor primary, use first available prediction
                if predictions:
                    first_pred = next(iter(predictions.values()))
                    predictions['fused'] = first_pred
                    logger.warning(f"Both 'fused' and 'primary' outputs missing, using fallback")

        return predictions

    def apply_confidence_router(self, logits_dict):
        """Apply confidence-based routing to improve fusion predictions"""
        import numpy as np
        import torch
        import torch.nn.functional as F

        # Convert numpy arrays to torch tensors for processing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logits = {}
        for name in logits_dict:
            # logits_dict contains accumulated softmax probs, need to convert back to logits
            # Since we have probabilities, we'll work with them directly
            logits[name] = torch.from_numpy(logits_dict[name]).to(device)

        if 'fused' not in logits:
            return logits_dict

        # Get probabilities (they're already probabilities from accumulation)
        probs = {}
        for name in ('primary', 'thermal', 'uv'):
            if name in logits:
                # Normalize the accumulated values to get probabilities
                p = logits[name]
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                probs[name] = p

        fused = logits['fused']
        tau = getattr(config, 'eval_conf_router_tau', 0.92)
        alpha = getattr(config, 'eval_conf_router_alpha', 0.8)

        # Max prob and class per head
        maxp = {}
        argm = {}
        for name in probs:
            maxp[name], argm[name] = probs[name].max(dim=-1)

        # Confident mask for each head
        sel = {name: (maxp[name] > tau) for name in probs}

        # Optional: class-specific routing (e.g., Onion->primary)
        routing = getattr(config, 'eval_router_class_to_mod', {})
        if routing:
            for cid, prefer in routing.items():
                if prefer in probs and prefer in logits:
                    m = (argm[prefer] == cid) & sel[prefer]
                    # Hard switch to preferred head logits for these pixels
                    if m.any():
                        fused[m] = logits[prefer][m]

        # Global blend for any confident head
        if sel:
            # Stack confidence values and find most confident head
            conf_vals = torch.stack([maxp[name] for name in sel], dim=0)
            conf_idx = conf_vals.argmax(dim=0)
            m_any = conf_vals.max(dim=0)[0] > tau

            if m_any.any():
                # Blend: fused = (1-alpha)*fused + alpha*head
                names = list(sel.keys())
                for i, name in enumerate(names):
                    pick = (conf_idx == i) & m_any
                    if pick.any() and name in logits:
                        fused[pick] = (1.0 - alpha) * fused[pick] + alpha * logits[name][pick]

        # Convert back to numpy and update dictionary
        logits_dict['fused'] = fused.cpu().numpy()

        return logits_dict

    def compute_multihead_metrics(self, predictions, label_primary, label_thermal, label_uv):
        """Compute metrics for each decoder head"""
        results_dict = {}

        # Check if we should skip individual head evaluations
        skip_individual = getattr(config, 'skip_individual_head_logits', False)

        if not skip_individual:
            # Primary decoder metrics
            if 'primary' in predictions and label_primary is not None:
                hist_p, labeled_p, correct_p = hist_info(config.num_classes, predictions['primary'], label_primary)
                results_dict['primary'] = {'hist': hist_p, 'labeled': labeled_p, 'correct': correct_p}

            # Thermal decoder metrics
            if 'thermal' in predictions and label_thermal is not None:
                num_classes_thermal = getattr(config, 'num_classes_thermal', config.num_classes)
                hist_t, labeled_t, correct_t = hist_info(num_classes_thermal, predictions['thermal'], label_thermal)
                results_dict['thermal'] = {'hist': hist_t, 'labeled': labeled_t, 'correct': correct_t}

            # UV decoder metrics
            if 'uv' in predictions and label_uv is not None:
                num_classes_uv = getattr(config, 'num_classes_uv', config.num_classes)
                hist_u, labeled_u, correct_u = hist_info(num_classes_uv, predictions['uv'], label_uv)
                results_dict['uv'] = {'hist': hist_u, 'labeled': labeled_u, 'correct': correct_u}

        return results_dict

    def compute_multihead_fusion_metrics(self, predictions, label_primary, label_thermal, label_uv):
        """Compute metrics for multi-head fusion (includes individual heads + fused output)"""
        results_dict = {}

        # Check if we should skip individual head evaluations
        skip_individual = getattr(config, 'skip_individual_head_logits', False)

        if not skip_individual:
            # Primary decoder metrics
            if 'primary' in predictions and label_primary is not None:
                hist_p, labeled_p, correct_p = hist_info(config.num_classes, predictions['primary'], label_primary)
                results_dict['primary'] = {'hist': hist_p, 'labeled': labeled_p, 'correct': correct_p}

            # Thermal decoder metrics
            if 'thermal' in predictions and label_thermal is not None:
                num_classes_thermal = getattr(config, 'num_classes_thermal', config.num_classes)
                hist_t, labeled_t, correct_t = hist_info(num_classes_thermal, predictions['thermal'], label_thermal)
                results_dict['thermal'] = {'hist': hist_t, 'labeled': labeled_t, 'correct': correct_t}

            # UV decoder metrics
            if 'uv' in predictions and label_uv is not None:
                num_classes_uv = getattr(config, 'num_classes_uv', config.num_classes)
                hist_u, labeled_u, correct_u = hist_info(num_classes_uv, predictions['uv'], label_uv)
                results_dict['uv'] = {'hist': hist_u, 'labeled': labeled_u, 'correct': correct_u}

        # Fused output metrics (evaluated against primary GT) - always compute if available
        if 'fused' in predictions and label_primary is not None:
            hist_f, labeled_f, correct_f = hist_info(config.num_classes, predictions['fused'], label_primary)
            results_dict['fused'] = {'hist': hist_f, 'labeled': labeled_f, 'correct': correct_f}

        return results_dict

    def save_multihead_predictions(self, predictions, name):
        """Save predictions for each decoder head"""
        for head_name, pred in predictions.items():
            head_save_path = f"{self.save_path}_{head_name}"
            head_save_path_color = f"{self.save_path}_{head_name}_color"
            
            ensure_dir(head_save_path)
            ensure_dir(head_save_path_color)

            fn = name + '.png'

            # Save raw prediction
            cv2.imwrite(os.path.join(head_save_path, fn), pred.astype(np.uint8))

            # Save colored prediction
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            
            # Use appropriate number of classes for coloring
            if head_name == 'thermal':
                num_classes = getattr(config, 'num_classes_thermal', config.num_classes)
            elif head_name == 'uv':
                num_classes = getattr(config, 'num_classes_uv', config.num_classes)
            else:
                num_classes = config.num_classes
                
            class_colors = get_class_colors(num_classes)
            palette_list = list(np.array(class_colors).flat)
            palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(head_save_path_color, fn))

        logger.info(f'Saved multi-head predictions for {name}')

    def save_multihead_fusion_predictions(self, predictions, name):
        """Save predictions for multi-head fusion (includes individual heads + fused output)"""
        for head_name, pred in predictions.items():
            head_save_path = f"{self.save_path}_{head_name}"
            head_save_path_color = f"{self.save_path}_{head_name}_color"
            
            ensure_dir(head_save_path)
            ensure_dir(head_save_path_color)

            fn = name + '.png'

            # Save raw prediction
            cv2.imwrite(os.path.join(head_save_path, fn), pred.astype(np.uint8))

            # Save colored prediction
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            
            # Use appropriate number of classes for coloring
            if head_name == 'thermal':
                num_classes = getattr(config, 'num_classes_thermal', config.num_classes)
            elif head_name == 'uv':
                num_classes = getattr(config, 'num_classes_uv', config.num_classes)
            else:  # primary or fused
                num_classes = config.num_classes
                
            class_colors = get_class_colors(num_classes)
            palette_list = list(np.array(class_colors).flat)
            palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(head_save_path_color, fn))

        logger.info(f'Saved multi-head fusion predictions for {name} (includes fused output)')
    
    def save_results_to_file(self, text_results, structured_results, custom_results_dir=None):
        """Save evaluation results to text and json files in results subfolder"""
        # Create results directory (use custom path if provided, otherwise default)
        if custom_results_dir:
            # Convert relative path to absolute path
            if not os.path.isabs(custom_results_dir):
                results_dir = os.path.join(config.root_dir, custom_results_dir)
            else:
                results_dir = custom_results_dir
        else:
            results_dir = os.path.join(config.root_dir, 'results')
        ensure_dir(results_dir)
        
        # Create filenames based on config name
        config_name = mini_args.config  # Get the config name from command line args
        text_file = os.path.join(results_dir, f"{config_name}.txt")
        json_file = os.path.join(results_dir, f"{config_name}.json")
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        # Save text results
        with open(text_file, 'w') as f:
            # Write header with metadata
            f.write(f"# Evaluation Results for {config_name}\n")
            f.write(f"# Config: {config_name}\n")
            f.write(f"# Dataset: {config.dataset_name}\n")
            f.write(f"# Date: {timestamp}\n")
            f.write(f"# Results files: {config_name}.txt (this file), {config_name}.json\n")
            f.write("#" * 80 + "\n\n")
            
            # Write text results
            f.write(text_results)
        
        # Add metadata to structured results
        structured_results['metadata'] = {
            'config_name': config_name,
            'dataset': config.dataset_name,
            'timestamp': timestamp,
            'text_file': f"{config_name}.txt",
            'json_file': f"{config_name}.json",
            'num_classes': config.num_classes,
            'class_names': config.class_names,
            'image_size': f"{config.image_height}x{config.image_width}",
            'backbone': config.backbone,
            'decoder': config.decoder,
            'epochs_trained': getattr(config, 'nepochs', 'unknown'),
            'batch_size': getattr(config, 'batch_size', 'unknown')
        }
        
        # Save JSON results
        import json
        with open(json_file, 'w') as f:
            json.dump(structured_results, f, indent=2)
        
        logger.info(f"Results saved to: {text_file} and {json_file}")

    def compute_metric(self, results):
        # Check if multi-head fusion results
        if getattr(config, 'use_multi_head_fusion', False) and results and isinstance(list(results[0].values())[0], dict):
            # Multi-head fusion results
            return self.compute_multihead_fusion_metric(results)
        # Check if multi-head results
        elif getattr(config, 'use_multi_head_decoder', False) and results and isinstance(list(results[0].values())[0], dict):
            # Multi-head results
            return self.compute_multihead_metric(results)
        else:
            # Single-head results (backward compatibility)
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

    def compute_multihead_metric(self, results):
        """Compute metrics for multi-head decoder results"""
        head_metrics = {}
        
        # First, determine which heads actually have data by examining the results
        available_heads = set()
        for result in results:
            available_heads.update(result.keys())
        
        # Initialize metric accumulators only for heads that have data
        for head_name in available_heads:
            head_metrics[head_name] = {
                'hist': None,
                'correct': 0,
                'labeled': 0,
                'count': 0
            }
        
        # Accumulate metrics for each head
        for result in results:
            for head_name, head_result in result.items():
                if head_name in head_metrics:
                    if head_metrics[head_name]['hist'] is None:
                        # Initialize histogram with correct dimensions
                        if head_name == 'thermal':
                            num_classes = getattr(config, 'num_classes_thermal', config.num_classes)
                        elif head_name == 'uv':
                            num_classes = getattr(config, 'num_classes_uv', config.num_classes)
                        else:
                            num_classes = config.num_classes
                        head_metrics[head_name]['hist'] = np.zeros((num_classes, num_classes))
                    
                    head_metrics[head_name]['hist'] += head_result['hist']
                    head_metrics[head_name]['correct'] += head_result['correct']
                    head_metrics[head_name]['labeled'] += head_result['labeled']
                    head_metrics[head_name]['count'] += 1
        
        # Compute metrics for each head and format results
        result_lines = []
        structured_results = {}  # For structured saving
        
        # Define output order: uv, thermal, primary (no fused in multihead-only mode)
        head_order = ['uv', 'thermal', 'primary']
        
        # Process heads in specified order
        for head_name in head_order:
            if head_name in head_metrics and head_metrics[head_name]['count'] > 0:
                metrics = head_metrics[head_name]
                hist = metrics['hist']
                correct = metrics['correct']
                labeled = metrics['labeled']
                
                iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
                
                # Store structured results
                structured_results[head_name] = {
                    'iou': iou.tolist(),
                    'mean_iou': float(mean_IoU),
                    'freq_iou': float(freq_IoU),
                    'mean_pixel_acc': float(mean_pixel_acc),
                    'pixel_acc': float(pixel_acc)
                }
                
                # Use appropriate class names for each head
                class_names = config.class_names  # All heads use same class names for now
                
                result_line = f"\n=== {head_name.upper()} DECODER RESULTS ===\n"
                result_line += print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                        class_names, show_no_back=False, no_print=True)
                result_lines.append(result_line)
        
        # Print the results with headers
        final_result = '\n'.join(result_lines)
        print(final_result)

        # Save results to file (use custom results_dir if specified)
        custom_results_dir = getattr(args, 'results_dir', None)
        self.save_results_to_file(final_result, structured_results, custom_results_dir)

        return final_result

    def compute_multihead_fusion_metric(self, results):
        """Compute metrics for multi-head fusion decoder results (includes fused output)"""
        head_metrics = {}
        
        # First, determine which heads actually have data by examining the results
        available_heads = set()
        for result in results:
            available_heads.update(result.keys())
        
        # Initialize metric accumulators only for heads that have data
        for head_name in available_heads:
            head_metrics[head_name] = {
                'hist': None,
                'correct': 0,
                'labeled': 0,
                'count': 0
            }
        
        # Accumulate metrics for each head + fused
        for result in results:
            for head_name, head_result in result.items():
                if head_name in head_metrics:
                    if head_metrics[head_name]['hist'] is None:
                        # Initialize histogram with correct dimensions
                        if head_name == 'thermal':
                            num_classes = getattr(config, 'num_classes_thermal', config.num_classes)
                        elif head_name == 'uv':
                            num_classes = getattr(config, 'num_classes_uv', config.num_classes)
                        else:  # primary or fused
                            num_classes = config.num_classes
                        head_metrics[head_name]['hist'] = np.zeros((num_classes, num_classes))
                    
                    head_metrics[head_name]['hist'] += head_result['hist']
                    head_metrics[head_name]['correct'] += head_result['correct']
                    head_metrics[head_name]['labeled'] += head_result['labeled']
                    head_metrics[head_name]['count'] += 1
        
        # Compute metrics for each head and format results
        result_lines = []
        structured_results = {}  # For structured saving
        
        # Define output order: uv, thermal, primary, fused (as requested)
        head_order = ['uv', 'thermal', 'primary', 'fused']
        
        # Process heads in specified order
        for head_name in head_order:
            if head_name in head_metrics and head_metrics[head_name]['count'] > 0:
                metrics = head_metrics[head_name]
                hist = metrics['hist']
                correct = metrics['correct']
                labeled = metrics['labeled']
                
                iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
                
                # Store structured results
                structured_results[head_name] = {
                    'iou': iou.tolist(),
                    'mean_iou': float(mean_IoU),
                    'freq_iou': float(freq_IoU),
                    'mean_pixel_acc': float(mean_pixel_acc),
                    'pixel_acc': float(pixel_acc)
                }
                
                # Use appropriate class names for each head
                class_names = config.class_names  # All heads use same class names for now
                
                header = f"\n=== {head_name.upper()} DECODER RESULTS"
                if head_name == 'fused':
                    header += " (FINAL FUSED OUTPUT)"
                header += " ==="
                result_line = header + "\n"
                result_line += print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                        class_names, show_no_back=False, no_print=True)
                result_lines.append(result_line)
        
        # Add summary comparison between primary and fused
        if 'primary' in head_metrics and 'fused' in head_metrics and head_metrics['fused']['count'] > 0:
            primary_hist = head_metrics['primary']['hist']
            fused_hist = head_metrics['fused']['hist']
            
            if primary_hist is not None and fused_hist is not None:
                _, primary_mean_iou, _, _, _, _ = compute_score(primary_hist, head_metrics['primary']['correct'], head_metrics['primary']['labeled'])
                _, fused_mean_iou, _, _, _, _ = compute_score(fused_hist, head_metrics['fused']['correct'], head_metrics['fused']['labeled'])
                
                improvement = fused_mean_iou - primary_mean_iou
                result_lines.append(f"\n=== FUSION IMPROVEMENT ===")
                result_lines.append(f"Primary mIoU: {primary_mean_iou:.4f}")
                result_lines.append(f"Fused mIoU: {fused_mean_iou:.4f}")
                result_lines.append(f"Improvement: {improvement:+.4f} ({improvement/primary_mean_iou*100:+.2f}%)")
                
                # Add to structured results
                structured_results['fusion_improvement'] = {
                    'primary_miou': float(primary_mean_iou),
                    'fused_miou': float(fused_mean_iou),
                    'improvement': float(improvement),
                    'improvement_percent': float(improvement/primary_mean_iou*100) if primary_mean_iou > 0 else 0.0
                }
        
        # Print the results with headers
        final_result = '\n'.join(result_lines)
        print(final_result)

        # Save results to file (use custom results_dir if specified)
        custom_results_dir = getattr(args, 'results_dir', None)
        self.save_results_to_file(final_result, structured_results, custom_results_dir)

        return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[mini_parser])
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--rgb_variant', type=str, default=None,
                        help='Specific RGB variant to evaluate (e.g., "3" for RGB3). Only used in RGBX mode.')

    # Normalization method flags
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Fast evaluation mode: skip individual head logits (multi-head fusion only), defaults to GN-16')
    parser.add_argument('--gn8', action='store_true', default=False,
                        help='Use GroupNorm with 8 groups for pyramid fusion')
    parser.add_argument('--gn16', action='store_true', default=False,
                        help='Use GroupNorm with 16 groups for pyramid fusion')
    parser.add_argument('--gn32', action='store_true', default=False,
                        help='Use GroupNorm with 32 groups for pyramid fusion')
    parser.add_argument('--ln', action='store_true', default=False,
                        help='Use LayerNorm for pyramid fusion (default for full mode)')

    # Shift evaluation overrides
    parser.add_argument('--results_dir', '-r', type=str, default=None,
                        help='Override results directory (default: ./results/)')
    parser.add_argument('--thermal_modality', type=str, default=None,
                        help='Override thermal modality folder (e.g., "LWIR_20PXD")')
    parser.add_argument('--uv_modality', type=str, default=None,
                        help='Override UV modality folder (e.g., "UV_8_20PXU")')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    # Handle RGBX mode and specific RGB variant selection
    if hasattr(config, 'rgb_mode') and config.rgb_mode == 'RGBX' and args.rgb_variant:
        # Override to evaluate only specific RGB variant
        logger.info(f"RGBX mode: Evaluating only RGB variant {args.rgb_variant}")
        original_variants = config.rgb_variants
        config.rgb_variants = [args.rgb_variant]
        # Update num_eval_imgs accordingly (divide by original number of variants)
        config.num_eval_imgs = config.num_eval_imgs // len(original_variants)

    # ============================================================================
    # Normalization configuration
    # ============================================================================
    # Count how many normalization flags were specified
    norm_flags_count = sum([args.gn8, args.gn16, args.gn32, args.ln])

    if norm_flags_count > 1:
        raise ValueError(
            "Cannot specify multiple normalization flags simultaneously. "
            "Choose only ONE of: --ln, --gn8, --gn16, --gn32"
        )

    # Apply --fast mode optimizations if requested
    if args.fast:
        logger.info("=" * 80)
        logger.info("FAST EVALUATION MODE ENABLED")
        logger.info("=" * 80)

        # For multi-head fusion models, skip individual head logits
        if hasattr(config, 'use_multi_head_fusion') and config.use_multi_head_fusion:
            config.skip_individual_head_logits = True
            logger.info("→ Multi-head fusion model: Skipping individual head logits (primary/thermal/uv)")
            logger.info("  Only computing fused output for speed")

        # Determine normalization type (explicit flag overrides default)
        if args.ln:
            config.prelogit_pyramid_norm = 'ln'
            logger.info("→ Using LayerNorm (overriding default GN-16)")
        elif args.gn8:
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 8
            logger.info("→ Using GroupNorm with 8 groups (overriding default GN-16)")
        elif args.gn32:
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 32
            logger.info("→ Using GroupNorm with 32 groups (overriding default GN-16)")
        else:
            # Default for --fast: GN-16
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 16
            logger.info("→ Using GroupNorm with 16 groups (default for fast mode)")

        logger.info("=" * 80 + "\n")

    # Apply normalization flags without --fast mode (just change norm, no skip)
    elif norm_flags_count > 0:
        logger.info("=" * 80)
        logger.info("NORMALIZATION OVERRIDE (Full Evaluation Mode)")
        logger.info("=" * 80)

        if args.ln:
            config.prelogit_pyramid_norm = 'ln'
            logger.info("→ Using LayerNorm")
        elif args.gn8:
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 8
            logger.info("→ Using GroupNorm with 8 groups")
        elif args.gn16:
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 16
            logger.info("→ Using GroupNorm with 16 groups")
        elif args.gn32:
            config.prelogit_pyramid_norm = 'gn'
            config.pyramid_norm_groups = 32
            logger.info("→ Using GroupNorm with 32 groups")

        logger.info("→ Full evaluation: Computing all head outputs")
        logger.info("=" * 80 + "\n")

    # ============================================================================
    # Modality override for shift evaluation
    # ============================================================================
    if args.thermal_modality or args.uv_modality:
        logger.info("=" * 80)
        logger.info("MODALITY OVERRIDE FOR SHIFT EVALUATION")
        logger.info("=" * 80)

        # Get X_MODALITY_CONFIG from the loaded config
        X_MODALITY_CONFIG = cfg_mod.X_MODALITY_CONFIG

        # Override thermal modality path if specified
        if args.thermal_modality:
            for k, v in X_MODALITY_CONFIG.items():
                if v.get('head') == 'LWIR':
                    old_path = v['path']
                    v['path'] = os.path.join(config.dataset_path, args.thermal_modality)
                    logger.info(f"Thermal modality override:")
                    logger.info(f"  Old: {old_path}")
                    logger.info(f"  New: {v['path']}")

        # Override UV modality path if specified
        if args.uv_modality:
            for k, v in X_MODALITY_CONFIG.items():
                if v.get('head') == 'UV':
                    old_path = v['path']
                    v['path'] = os.path.join(config.dataset_path, args.uv_modality)
                    logger.info(f"UV modality override:")
                    logger.info(f"  Old: {old_path}")
                    logger.info(f"  New: {v['path']}")

        # Rebuild x_root_folders list with updated paths
        config.x_root_folders = []
        config.x_formats = []
        config.x_is_single_channel = []
        config.x_align_needed = []
        config.x_decoder_heads = []

        multi_chan_keys = ("DIN", "LWIR", "RGB0_")
        for k in config.active_modalities():
            entry = X_MODALITY_CONFIG[k]
            config.x_root_folders.append(entry["path"])
            config.x_formats.append(".png")  # Use hardcoded .png format

            folder_name = os.path.basename(entry["path"])
            is_single = not any(folder_name.startswith(mc) for mc in multi_chan_keys)
            config.x_is_single_channel.append(is_single)
            config.x_align_needed.append(entry.get("align", False))
            config.x_decoder_heads.append(entry.get("head", "main"))

        logger.info("Rebuilt x_root_folders:")
        for i, folder in enumerate(config.x_root_folders):
            logger.info(f"  [{i}] {folder} (head: {config.x_decoder_heads[i]})")
        logger.info("=" * 80 + "\n")

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
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
        'eval_source': config.eval_source,
        # multi-GT configuration
        'multi_gt_enabled': getattr(config, 'multi_gt_enabled', False),
        'gt_thermal_root': getattr(config, 'gt_thermal_folder', ''),
        'gt_uv_root': getattr(config, 'gt_uv_folder', ''),
        'gt_thermal_format': getattr(config, 'gt_thermal_format', '.png'),
        'gt_uv_format': getattr(config, 'gt_uv_format', '.png')
    }

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