import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.transforms import pad_image_to_shape, normalize
from config import config as C

logger = get_logger()


class Evaluator(object):
    def __init__(self, dataset, class_num, norm_mean, norm_std, network, multi_scales, 
                is_flip, devices, verbose=False, save_path=None, show_image=False, stopatfirst=False):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image
        
        self.stopatfirst = stopatfirst #to test and break after first image
        self.nclass = C.num_classes
        self.class_names = C.class_names

    def run(self, model_path, model_indice, log_file, log_file_link, stopatfirst=False):
        self.stopatfirst = stopatfirst
        
        log_dir = os.path.dirname(log_file)
        if log_dir:
            ensure_dir(log_dir)
            
        results = open(log_file, 'a')
        link_file(log_file, log_file_link)
    
        # Debugging: Print or log the input arguments
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Model indice: {model_indice}")
        logger.debug(f"Log file: {log_file}")
        logger.debug(f"Log file link: {log_file_link}")
    
        # Add debugging information to understand the flow
        if '.pth' in model_indice:
            models = [model_indice, ]
            logger.debug(f"Single model file specified: {models}")
        elif "-" in model_indice:
            # Debugging: Print the models list after processing
            logger.debug(f"Models list after processing range: {models}")
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path, 'epoch-%s.pth' % model_indice), ]
                logger.debug(f"Model path exists. Models list: {models}")
            else:
                models = [None]
                logger.debug("Model path does not exist. Models set to [None]")
        
        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            if len(self.devices ) == 1:
                result_line = self.single_process_evalutation()
            else:
                result_line = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()


    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()

        logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd,self.devices[0])
            all_results.append(results_dict)
            
            # Break after the first image if stopatfirst is True
            if self.stopatfirst:
                break
        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line


    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):

            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info('GPU %s handle %d data.' % (device, len(shred_list)))

            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device))
            procs.append(p)

        for p in procs:

            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (
            device, time.time() - start_load_time))

        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device)
            self.results_queue.put(results_dict)

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError

    # evaluate the whole image at once
    def whole_eval(self, img, output_size, device=None):
        processed_pred = np.zeros(
            (output_size[0], output_size[1], self.class_num))

        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_LINEAR)
            scaled_img = self.process_image(scaled_img, None)
            pred = self.val_func_process(scaled_img, device)
            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(pred.cpu().numpy(),
                                         (output_size[1], output_size[0]),
                                         interpolation=cv2.INTER_LINEAR)

        pred = processed_pred.argmax(2)

        return pred

    # slide the window to evaluate the image
    def sliding_eval(self,
                     img: np.ndarray,
                     modal_xs: Optional[List[np.ndarray]],
                     crop_size: Tuple[int,int],
                     stride_rate: float,
                     device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Generic sliding‐window evaluation for RGB + (optional) extra modalities.
        img:       H×W×3 RGB image (np.uint8 or float32)
        modal_xs:  None or list of H×W×C_i extra‐modality maps
        crop_size: (h, w) patch size for sliding window
        stride_rate: fraction of patch size to stride by
        device:    torch device for inference
        Returns:   H×W  argmaxed prediction map
        """
        ori_h, ori_w, _ = img.shape
        accum = np.zeros((ori_h, ori_w, self.class_num), dtype=np.float32)

        # for each scale
        for s in self.multi_scales:
            # 1) resize RGB
            img_s = cv2.resize(img,
                               None,
                               fx=s, fy=s,
                               interpolation=cv2.INTER_LINEAR)

            # 2) resize each extra modality (if present)
            if modal_xs is None:
                # pure‐RGB path
                accum += self.scale_process(
                    img_s,
                    (ori_h, ori_w),
                    crop_size,
                    stride_rate,
                    device
                )
            else:
                # RGB+X path
                scaled_mods = []
                for mx in modal_xs:
                    m = mx
                    # ensure 3‐d array
                    if m.ndim == 2:
                        m = m[:, :, None]
                    # nearest for single‐channel, bilinear for multi
                    interp = (cv2.INTER_NEAREST
                              if m.shape[2] == 1
                              else cv2.INTER_LINEAR)
                    m_s = cv2.resize(m,
                                     None,
                                     fx=s, fy=s,
                                     interpolation=interp)
                    scaled_mods.append(m_s)

                # stack them back into one H×W×C_total array
                mod_comb = np.concatenate(scaled_mods, axis=2)

                accum += self.scale_process_rgbX(
                    img_s,
                    mod_comb,
                    (ori_h, ori_w),
                    crop_size,
                    stride_rate,
                    device
                )

        # final argmax over classes
        return accum.argmax(axis=2)



    def scale_process(self, img, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data)
                score = score[0]

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                # score = torch.exp(score)
                # score = score.data

        return score

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.norm_mean, self.norm_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img

    
    # add new function for rgb and modal X segmentation
    def sliding_eval_rgbX(self,
                          img,
                          modal_x,
                          crop_size,
                          stride_rate,
                          device=None):
        """
        Multi-modal sliding-window evaluation (RGB + extra modalities).
        img         : H×W×3 numpy array or torch.Tensor (C×H×W)
        modal_x     : list of H×W×C_m numpy arrays, OR torch.Tensor (1×C_total×H×W)
        crop_size   : (h, w) or int
        stride_rate : float
        device      : cuda device index or None
        """
        # 1) Convert RGB to H×W×3 numpy if needed
        if isinstance(img, torch.Tensor):
            # assume shape (C, H, W) or (1, C, H, W)
            arr = img.cpu().numpy()
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            img = arr.transpose(1, 2, 0)

        # 2) Convert modal_x to list of H×W×C_m numpy arrays
        if isinstance(modal_x, torch.Tensor):
            mx = modal_x.cpu().numpy()
            if mx.ndim == 4 and mx.shape[0] == 1:
                mx = mx[0]           # now C_total×H×W
            # split channels back into per-modality
            ch_counts = [1 if s else 3 for s in C.x_is_single_channel]  # depth/intensity=1, others=3 :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
            splits = np.split(mx, np.cumsum(ch_counts)[:-1], axis=0)
            modal_list = [sp.transpose(1, 2, 0) for sp in splits]
        else:
            # assume already a list of numpy arrays
            modal_list = modal_x

        # 3) Prepare output accumulator
        ori_h, ori_w, _ = img.shape
        processed_pred = np.zeros((ori_h, ori_w, self.class_num), dtype=np.float32)

        # 4) Multi-scale loop
        for s in self.multi_scales:
            new_h, new_w = int(ori_h * s), int(ori_w * s)
            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # resize each extra modality
            if modal_list is not None:
                modal_scale = []
                for m in modal_list:
                    interp = cv2.INTER_NEAREST if m.ndim == 2 else cv2.INTER_LINEAR
                    modal_scale.append(
                        cv2.resize(m, (new_w, new_h), interpolation=interp)
                    )
            else:
                modal_scale = None

            # 5) call your rgbX processor
            processed_pred += self.scale_process_rgbX(
                img_scale,
                modal_scale,
                (ori_h, ori_w),
                to_2tuple(crop_size),
                stride_rate,
                device
            )

        # 6) final argmax
        return processed_pred.argmax(axis=2)

    def scale_process_rgbX(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
            input_data, input_modal_x, margin = self.process_image_rgbX(img, modal_x, crop_size)
            score = self.val_func_process_rgbX(input_data, input_modal_x, device) 
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0]
                    s_y = grid_yidx * stride[1]
                    e_x = min(s_x + crop_size[0], pad_cols)
                    e_y = min(s_y + crop_size[1], pad_rows)
                    s_x = e_x - crop_size[0]
                    s_y = e_y - crop_size[1]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    if len(modal_x_pad.shape) == 2:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x]
                    else:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x,:]

                    input_data, input_modal_x, tmargin = self.process_image_rgbX(img_sub, modal_x_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, input_modal_x, device)
                    
                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                            tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process_rgbX(self, input_data, input_modal_x, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
    
        input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)
    
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data, input_modal_x)
                score = score[0]
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_modal_x = input_modal_x.flip(-1)
                    score_flip = self.val_func(input_data, input_modal_x)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)
        
        return score

    # for rgbd segmentation
    def process_image_rgbX(self, img, modal_x_list, crop_size=None):
        # Normalize RGB exactly as before...
        p_img = img
        if p_img.shape[2] < 3:
            p_img = np.repeat(p_img, 3, axis=2)
        p_img = normalize(p_img, self.norm_mean, self.norm_std)
    
        # Now handle the list of modality arrays:
        normalized_mods = []
        for arr, is_single in zip(modal_x_list, C.x_is_single_channel):
            # arr has shape (H, W) or (H, W, C)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            # apply the correct norm stats per‐channel
            # for simplicity use C.x_norm_stats to pick mean/std
            idx = len(normalized_mods)
            mean = C.x_norm_stats[idx]['mean']
            std  = C.x_norm_stats[idx]['std']
            if len(mean) != arr.shape[2]:
                mean = [mean[0]] * arr.shape[2]
                std  = [std[0]]  * arr.shape[2]
            normed = normalize(arr, mean, std)
            normalized_mods.append(normed)
    
        # Concatenate back to a single array before padding/transposing
        p_modal_x = np.concatenate(normalized_mods, axis=2)
           
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, 0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, 0)
            if p_modal_x.ndim == 2:
                p_modal_x = p_modal_x[:, :, None]
            return p_img.transpose(2,0,1), p_modal_x.transpose(2,0,1), margin
    
        # final no‐crop case
        return p_img.transpose(2,0,1), p_modal_x.transpose(2,0,1)
