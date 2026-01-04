import os
import os.path as osp
import sys
import time
import torch
import numpy as np
from easydict import EasyDict as edict
import argparse

from engine.logger import get_logger
import logging

logger = get_logger()

C = edict()
config = C
cfg = C

C.seed = 12345


remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.data_dir = "/path-to-data" 
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'GFNET'
C.dataset_path = osp.join(C.data_dir, 'datasets', 'MM5_FULL')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB3') #RGB
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'ANNO_CLASS') # GT
C.gt_format = '.png'
C.gt_transform = False #MB: Deducts 1 from GT value if set to True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.MM5Dataset
# True for most dataset valid, Faslse for MFNet(?)
# C.x_root_folder = osp.join(C.dataset_path, 'T24') # 2nd modality # HHA rawDepthsFocus depths3CH depths depthsIP rawDepthsFocus980N2 rawDepthsFocus980N

# ---------------------------------------------------------------
#                       EXTRA MODALITIES
# ---------------------------------------------------------------

DATA_ROOT = C.dataset_path

# For each possible extra stream, we now carry both:
#   - `path`   : where to find the images (empty = disabled)
#   - `align`  : whether to run alignment (STN/FRM/TPS) on this stream
X_MODALITY_CONFIG = {
    1: { "path": osp.join(DATA_ROOT, "DIN"),          "align": False },
    2: { "path": osp.join(DATA_ROOT, "T24"),          "align": True },
    3: { "path": osp.join(DATA_ROOT, "U8"),           "align": True },
    4: { "path": "",  "align": False },
}

# X_MODALITY_CONFIGx = {
#     1: { "path": osp.join(DATA_ROOT, "IAIP"),         "align": False },
#     2: { "path": osp.join(DATA_ROOT, "D_FocusN"),     "align": False },
#     3: { "path": osp.join(DATA_ROOT, "T24"),          "align": True  },
#     4: { "path": osp.join(DATA_ROOT, "U8"),           "align": True },
# }

def active_modalities():
    return sorted(k for k,c in X_MODALITY_CONFIG.items() if c["path"])

C.active_modalities = active_modalities
C.n_modal = len(C.active_modalities())
# now build the per‐modality lists
C.x_root_folders      = []
C.x_formats           = []
C.x_is_single_channel = []
C.x_align_needed      = []

multi_chan_keys = ("U8x", "HHA", "DIM", "DII", "DIN", "3CH", "T24", "980", "F_")
for k in active_modalities():
    entry = X_MODALITY_CONFIG[k]
    root = entry["path"]
    C.x_root_folders.append(root)
    C.x_formats.append(".png")

    # single–vs–multi channel detection
    is_single = not any(substr in root for substr in multi_chan_keys)
    C.x_is_single_channel.append(is_single)

    # alignment flag for that stream
    C.x_align_needed.append(entry["align"])

# total depth of extra channels (sum 1 or 3 each)
C.x_in_chans = sum(1 if s else 3 for s in C.x_is_single_channel)

# make a short tag for your experiment name
modal_basenames = [osp.basename(p) for p in C.x_root_folders]
modality_tag    = "-".join(modal_basenames) if modal_basenames else "noX"
C.modalities_tag = modality_tag

# ---------------------------------------------------------------

# Dynamically append folder names to dataset_name
# C.dataset_name = f"{C.dataset_name}-{osp.basename(C.rgb_root_folder)}-{modality_tag}"

# C.x_format = '.png'
# if 'HHA' in C.x_root_folder or '3CH' in C.x_root_folder or 'T24' in C.x_root_folder or '980' in C.x_root_folder or 'F_' in C.x_root_folder:
#     C.x_is_single_channel = False
# else:
#     C.x_is_single_channel = True
#C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "list_train_f.txt")
C.eval_source = osp.join(C.dataset_path, "list_eval_f.txt")
C.is_test = False
C.num_train_imgs = 173 #795
C.num_eval_imgs = 44 #654
# C.num_classes = 34
# C.class_names =  ['Background', 'Lemon', 'Lemon Bad', 'Lemon Fake', 'Mirror', 'Bowl', 'Mandarin', 'Mandarin Bad', 'Mandarin Fake', 'Kettle', 'Lemon Half', 'Lemon Sliced', 'Mandarin Half', 'Mandarin Sliced', 'Mandarin Peel', 'Cup Hot', 'Onion Red', 'Onion', 'Grapes Green', 'Grapes Green Bad', 'Grapes Green Fake', 'Grapes Blue Fake', 'Grapes Blue', 'Grapes Blue Bad', 'Apple', 'Apple Fake', 'Apple Green', 'Apple Green Bad', 'Apple Green Fake', 'Cup Cold', 'Pear', 'Pear Bad', 'Carrot', 'Carrot Fake']
# C.class_names2IDs = {'Bkg': 0, 'Lemon': 1, 'Lemon Bad': 2, 'Lemon Fake': 3, 'Mirror': 4, 'Bowl': 5, 'Mandarin': 6, 'Mandarin Bad': 7, 'Mandarin Fake': 8, 'Kettle': 9, 'Lemon Half': 10, 'Lemon Sliced': 11, 'Mandarin Half': 12, 'Mandarin Sliced': 13, 'Mandarin Peel': 14, 'Cup Hot': 15, 'Onion Red': 16, 'Onion': 17, 'Grapes Green': 18, 'Grapes Green Bad': 19, 'Grapes Green Fake': 20, 'Grapes Blue Fake': 21, 'Grapes Blue': 22, 'Grapes Blue Bad': 23, 'Apple': 24, 'Apple Fake': 25, 'Apple Green': 26, 'Apple Green Bad': 27, 'Apple Green Fake': 28, 'Cup Cold': 29, 'Pear': 30, 'Pear Bad': 31, 'Carrot': 32, 'Carrot Fake': 33}
# fixed
# C.num_classes = 33
# C.class_names = ['Background', 'Lemon', 'Lemon Bad', 'Lemon Fake', 'Mirror', 'Bowl', 'Mandarin', 'Mandarin Bad', 'Mandarin Fake', 'Kettle', 'Lemon Half', 'Mandarin Half', 'Mandarin Sliced', 'Mandarin Peel', 'Cup Hot', 'Onion Red', 'Onion', 'Grapes Green', 'Grapes Green Bad', 'Grapes Green Fake', 'Grapes Blue Fake', 'Grapes Blue', 'Grapes Blue Bad', 'Apple', 'Apple Fake', 'Apple Green', 'Apple Green Bad', 'Apple Green Fake', 'Cup Cold', 'Pear', 'Pear Bad', 'Carrot', 'Carrot Fake']
# C.class_names2IDs = {'Bkg': 0, 'Lemon': 1, 'Lemon Bad': 2, 'Lemon Fake': 3, 'Mirror': 4, 'Bowl': 5, 'Mandarin': 6, 'Mandarin Bad': 7, 'Mandarin Fake': 8, 'Kettle': 9, 'Lemon Half': 10, 'Mandarin Half': 11, 'Mandarin Sliced': 12, 'Mandarin Peel': 13, 'Cup Hot': 14, 'Onion Red': 15, 'Onion': 16, 'Grapes Green': 17, 'Grapes Green Bad': 18, 'Grapes Green Fake': 19, 'Grapes Blue Fake': 20, 'Grapes Blue': 21, 'Grapes Blue Bad': 22, 'Apple': 23, 'Apple Fake': 24, 'Apple Green': 25, 'Apple Green Bad': 26, 'Apple Green Fake': 27, 'Cup Cold': 28, 'Pear': 29, 'Pear Bad': 30, 'Carrot': 31, 'Carrot Fake': 32}
C.num_classes = 32
C.class_names = ['Background', 'Lemon', 'Lemon Bad', 'Lemon Fake', 'Mirror', 'Bowl', 'Mandarin', 'Mandarin Bad', 'Mandarin Fake', 'Kettle', 'Lemon Half', 'Mandarin Half', 'Mandarin Peel', 'Cup Hot', 'Onion Red', 'Onion', 'Grapes Green', 'Grapes Green Bad', 'Grapes Green Fake', 'Grapes Blue Fake', 'Grapes Blue', 'Grapes Blue Bad', 'Apple', 'Apple Fake', 'Apple Green', 'Apple Green Bad', 'Apple Green Fake', 'Cup Cold', 'Pear', 'Pear Bad', 'Carrot', 'Carrot Fake']
C.class_names2IDs = {'Bkg': 0, 'Lemon': 1, 'Lemon Bad': 2, 'Lemon Fake': 3, 'Mirror': 4, 'Bowl': 5, 'Mandarin': 6, 'Mandarin Bad': 7, 'Mandarin Fake': 8, 'Kettle': 9, 'Lemon Half': 10, 'Mandarin Half': 11, 'Mandarin Peel': 12, 'Cup Hot': 13, 'Onion Red': 14, 'Onion': 15, 'Grapes Green': 16, 'Grapes Green Bad': 17, 'Grapes Green Fake': 18, 'Grapes Blue Fake': 19, 'Grapes Blue': 20, 'Grapes Blue Bad': 21, 'Apple': 22, 'Apple Fake': 23, 'Apple Green': 24, 'Apple Green Bad': 25, 'Apple Green Fake': 26, 'Cup Cold': 27, 'Pear': 28, 'Pear Bad': 29, 'Carrot': 30, 'Carrot Fake': 31}
"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
 # Tiny
# C.image_height = 240
# C.image_width  = 320

# ---------------------------------------------------------------
#                   PER-MODALITY NORMALIZATION STATS (0-1 Range)
# ---------------------------------------------------------------

# --- Calculated Dataset Statistics (all values directly in 0-1 range) ---
# Paste all calculated mean and std values (scaled to 0-1) here.
ALL_MODALITY_NORM_STATS_0_1 = {
    "U8": {'mean': [0.336560515434], 'std': [0.242863837856], 'expected_load_channels': 1},
    "U8x": {'mean': [0.336560515434]*3, 'std': [0.242863837856]*3, 'expected_load_channels': 3},
    "T24": {'mean': [0.684933897286, 0.773390145426, 0.264115819698], 'std': [0.106936764157, 0.148027290293, 0.133550142102], 'expected_load_channels': 3},
    "DIM": {'mean': [0.579844858158, 0.405508030459, 0.013400419702], 'std': [0.335593713285, 0.173974247617, 0.032409018686], 'expected_load_channels': 3},
    "DIN": {'mean': [0.579844858158, 0.405508030459, 0.382099498189], 'std': [0.335593713285, 0.173974247617, 0.090787469763], 'expected_load_channels': 3},
    "DII": {'mean': [0.579844858158, 0.405508030459, 0.405508030459], 'std': [0.335593713285, 0.173974247617, 0.173974247617], 'expected_load_channels': 3},
    "IAIP": {'mean': [0.405492278671], 'std': [0.1739035604], 'expected_load_channels': 1},
    "D_Focus": {'mean': [0.57990113071], 'std': [0.335521732019], 'expected_load_channels': 1},
    "D_FocusN": {'mean': [0.544847478399], 'std': [0.220548644209], 'expected_load_channels': 1},
    "RGB1": {'mean': [0.001774950845, 0.029499471925, 0.070820739436], 'std': [0.005856136616, 0.042310220468, 0.14436700879], 'calculated_channels': 3},
    "RGB2": {'mean': [0.046487510748, 0.308682168621, 0.212374398208], 'std': [0.083473167951, 0.148903374621, 0.300307732587], 'calculated_channels': 3},
    "RGB3": {'mean': [0.179796032672, 0.569485112958, 0.341689441857], 'std': [0.132387636081, 0.170978688671, 0.330585026916], 'calculated_channels': 3},
    "RGB4": {'mean': [0.017520847044, 0.160501330175, 0.222885552556], 'std': [0.057634461422, 0.174199797041, 0.269092502016], 'calculated_channels': 3},
    "RGB5": {'mean': [0.721418016818, 0.947966706571, 0.894091304536], 'std': [0.230738598908, 0.092929617109, 0.166237558185], 'calculated_channels': 3},
    "RGB6": {'mean': [0.048193568351, 0.049485948075, 0.120062169357], 'std': [0.111998312979, 0.06210439709, 0.194498443369], 'calculated_channels': 3},
    "RGB7": {'mean': [0.013386141583, 0.211926209565, 0.231888663759], 'std': [0.038659881191, 0.151018923636, 0.266900294135], 'calculated_channels': 3},
    "RGB8": {'mean': [0.010397139199, 0.083173501265, 0.093672163091], 'std': [0.016244231091, 0.050034977768, 0.172608812093], 'calculated_channels': 3}
}

# --- Dynamic setting of Primary RGB Normalization (0-1 Range) ---
primary_rgb_folder_basename = None
if hasattr(C, 'rgb_root_folder') and C.rgb_root_folder:
    primary_rgb_folder_basename = osp.basename(C.rgb_root_folder)

# Look up the stats for the primary RGB folder
rgb_norm_stats_0_1 = ALL_MODALITY_NORM_STATS_0_1.get(primary_rgb_folder_basename)

if rgb_norm_stats_0_1:
    C.norm_mean = np.array(rgb_norm_stats_0_1['mean'], dtype=np.float32)
    C.norm_std = np.array(rgb_norm_stats_0_1['std'], dtype=np.float32)
    logger.info(f"Using dataset-specific norm stats (0-1 range) for primary RGB: {primary_rgb_folder_basename}")
    logger.info(f"  Mean (0-1): {C.norm_mean.tolist()}")
    logger.info(f"  Std Dev (0-1): {C.norm_std.tolist()}")
else:
    logger.warning(f"No specific norm stats found for primary RGB folder: '{primary_rgb_folder_basename}'.")
    # Fallback if C.norm_mean/std are not already defined (e.g. by user or initial lines above)
    if not hasattr(C, 'norm_mean') or C.norm_mean is None:
        logger.info("Falling back to ImageNet normalization stats (0-1 range) for primary RGB.")
        C.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # ImageNet mean (0-1)
        C.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std (0-1)
    else:
        # C.norm_mean/std were already set, assume they are in 0-1 range and correct type.
        logger.info(f"Using pre-configured C.norm_mean/C.norm_std (assumed 0-1 range and np.float32) as fallback.")
        # Ensure they are float32 numpy arrays if they were set by other means
        C.norm_mean = np.array(C.norm_mean, dtype=np.float32)
        C.norm_std = np.array(C.norm_std, dtype=np.float32)


# --- Logic for populating C.x_norm_stats for Extra Modalities (0-1 Range) ---
C.x_norm_stats = []
active_mod_details_log = []

# Define default stats in 0-1 range
DEFAULT_NORM_1CH_0_1 = {'mean': [0.5], 'std': [0.5]}
DEFAULT_NORM_3CH_0_1 = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

USE_MM5_STATS = True
# Check if active_modalities and X_MODALITY_CONFIG are defined and runnable
# This part depends on your project's structure for defining these.
if callable(globals().get('active_modalities')) and \
   'X_MODALITY_CONFIG' in globals() and \
   hasattr(C, 'x_is_single_channel') and \
   len(C.x_is_single_channel) == len(active_modalities()):

    for i, mod_cfg_key in enumerate(active_modalities()):
        mod_config = X_MODALITY_CONFIG.get(mod_cfg_key)
        if not mod_config or "path" not in mod_config:
            logger.warning(f"Skipping modality with key '{mod_cfg_key}': Configuration missing or path not found.")
            # Append a placeholder or handle error appropriately
            C.x_norm_stats.append(DEFAULT_NORM_1CH_0_1.copy() if C.x_is_single_channel[i] else DEFAULT_NORM_3CH_0_1.copy()) # Fallback
            active_mod_details_log.append({
                'path': 'N/A_OR_ERROR',
                'model_expects_single_ch': C.x_is_single_channel[i],
                'norm_mean': C.x_norm_stats[-1]['mean'],
                'norm_std': C.x_norm_stats[-1]['std'],
                'status': 'Error/Config Missing'
            })
            continue

        mod_path_full = mod_config["path"]
        mod_basename = osp.basename(mod_path_full)
        model_expects_single_channel = C.x_is_single_channel[i]
        model_expected_channels = 1 if model_expects_single_channel else 3

        norm_stat_to_use = None
        # Stats from ALL_MODALITY_NORM_STATS_0_1 are already in 0-1 range
        mod_stats_from_all_0_1 = ALL_MODALITY_NORM_STATS_0_1.get(mod_basename)

        if mod_stats_from_all_0_1 and USE_MM5_STATS:
            mean_list_0_1 = mod_stats_from_all_0_1['mean'] # Already 0-1
            std_list_0_1 = mod_stats_from_all_0_1['std']   # Already 0-1

            current_mean_values = None
            current_std_values = None
            log_msg_prefix = "Using specific"

            if len(mean_list_0_1) == model_expected_channels:
                current_mean_values = mean_list_0_1
                current_std_values = std_list_0_1
                log_msg_prefix = f"Using specific {len(mean_list_0_1)}-ch"
            elif len(mean_list_0_1) == 1 and model_expected_channels == 3: # Broadcast 1ch to 3ch
                current_mean_values = mean_list_0_1 * 3
                current_std_values = std_list_0_1 * 3
                log_msg_prefix = f"Broadcasting 1-ch specific"
            elif len(mean_list_0_1) == 3 and model_expected_channels == 1: # Use first channel of 3ch for 1ch
                current_mean_values = mean_list_0_1[0:1]
                current_std_values = std_list_0_1[0:1]
                log_msg_prefix = f"Using first channel of 3-ch specific"
                logger.warning(f"Using only first channel of 3-ch stats for 1-ch modality '{mod_basename}'.")
            else:
                logger.warning(f"Channel count mismatch for extra modality '{mod_basename}' (Expected: {model_expected_channels}, Found: {len(mean_list_0_1)}). Falling back to default 0-1 stats.")

            if current_mean_values is not None and current_std_values is not None:
                norm_stat_to_use = {'mean': current_mean_values, 'std': current_std_values}
                logger.info(f"{log_msg_prefix} norm stats (0-1 range) for extra modality: {mod_basename}.")
                status_log = "Specific"
            else:
                status_log = "Fallback (Channel Mismatch)"

        else: # mod_stats_from_all_0_1 is None
            status_log = f"Fallback (Not in ALL_MODALITY_NORM_STATS_0_1)"


        if norm_stat_to_use is None: # Fallback to 0-1 range defaults
            logger.warning(f"No suitable specific norm_stats found for extra modality '{mod_basename}'. Using default (0-1 range).")
            if model_expects_single_channel:
                norm_stat_to_use = DEFAULT_NORM_1CH_0_1.copy()
            else:
                norm_stat_to_use = DEFAULT_NORM_3CH_0_1.copy()

        C.x_norm_stats.append(norm_stat_to_use)
        active_mod_details_log.append({
            'path': mod_path_full,
            'model_expects_single_ch': model_expects_single_channel,
            'norm_mean': norm_stat_to_use['mean'],
            'norm_std': norm_stat_to_use['std'],
            'status': status_log if 'status_log' in locals() and norm_stat_to_use is not None else "Fallback (Default)"
        })
        if 'status_log' in locals(): del status_log # clean up for next iteration


    logger.info("Active Extra Modality Normalization Configuration (0-1 Range):")
    for detail in active_mod_details_log:
        logger.info(f"  Path: {detail['path']}, ModelExpectsSingle: {detail['model_expects_single_ch']}, Mean: {detail['norm_mean']}, Std: {detail['norm_std']}, Status: {detail['status']}")
elif not callable(globals().get('active_modalities')):
    logger.info("`active_modalities` function not found. Skipping extra modality normalization setup.")
elif 'X_MODALITY_CONFIG' not in globals():
    logger.info("`X_MODALITY_CONFIG` not found. Skipping extra modality normalization setup.")
else:
    logger.info("Issue with `C.x_is_single_channel` or its length. Skipping extra modality normalization setup.")

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b0' # Backbone architecture
C.pretrained_model = None # Path to pretrained weights (e.g., './pretrained/mit_b2.pth') or None
C.decoder = 'MLPDecoder' # Decoder type
C.decoder_embed_dim = 256 #b2: 512 # Embedding dimension for MLP decoder
# C.decoder = 'deeplabv3+'
# C.aux_rate = 0.4
C.optimizer = 'AdamW'

"""Fusion Config"""
# Alignment method choices: "frm", "stn", "tps", "none"
C.alignment_method = "frm"
# Fusion method choices: "ffm", "cafb", "add", "stn" (stn uses FineRegistrationFusion)
C.fusion_method = "ffm" # not used when using hyper or intensity fusion

C.fusion_combination = "sigmoid_gating" # if sigmoid_gating - it replaces the fusion method!
 # sgate fusion options: add / avg 
C.sgate_fusion_mode="add" # sigmoid_gating fusion method
# Dedicated fusion options (set only one to True, or both False)
C.use_intensity_enhancement = True # Use stage-wise RGB+I fusion (StageWiseRGBIntensityFusion)

# <<< --- Loss Function Configuration --- >>>
# Options: 'CE', 'Dice', 'Focal', 'CEDice'
C.loss_function_name = 'CEDice'

# Parameters specific to loss functions
C.loss_params = edict()
# -- Focal Loss params --
C.loss_params.focal_gamma = 2.0
C.loss_params.focal_alpha = None # Set to list/tensor of weights per class, or None
# -- Dice Loss params --
C.loss_params.dice_smooth = 1.0
C.loss_params.dice_average = 'macro' # 'micro' or 'macro'
C.loss_params.dice_average_foreground = True # For macro average, average only foreground classes?
# -- CEDice params --
C.loss_params.cedice_ce_weight = 0.5 # Weight for CrossEntropy part
C.loss_params.cedice_dice_weight = 0.5 # Weight for Dice part

# Class weighting strategy for the chosen loss function
# Options: 'none', 'manual', 'inverse_freq' (inverse_freq requires implementation in train.py)
C.loss_class_weights_type = 'manual'
# Manual weights (list of length C.num_classes). Used if class_weights_type is 'manual'.
# Example: Lower weight for background (class 0)
weights = [1.0] * C.num_classes
weights[0]  = 0.1   # background or class-0
weights[11] = 15.0   # 'Mandarin Half'
weights[12] = 30.0   # 'Mandarin Peel'
C.loss_manual_class_weights = weights

# Rare class handling
C.rare_class_multiply = True
C.rare_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
C.rare_class_ids_factor = [1, 2, 2, 2, 2, 2, 2, 3, 6, 3, 3, 5, 3, 3, 3, 2, 3, 3, 4, 3, 4, 2, 3, 2, 3, 3, 1, 3, 3, 3, 3]
C.rare_files_ids = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '13', '14', '15', '18', '19', '20', '21', '22', '24', '26', '29', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '50', '53', '54', '55', '56', '58', '59', '60', '61', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '77', '78', '79', '81', '82', '83', '84', '86', '87', '88', '89', '90', '91', '93', '94', '95', '96', '99', '100', '102', '104', '105', '106', '107', '108', '110', '111', '112', '113', '114', '116', '117', '120', '121', '123', '124', '126', '127', '128', '130', '131', '132', '133', '135', '137', '139', '140', '142', '144', '145', '146', '147', '148', '149', '151', '152', '153', '154', '155', '156', '157', '158', '159', '161', '163', '165', '166', '167', '168', '169', '171', '173', '175', '176', '177', '179', '180', '181', '182', '183', '184', '186', '187', '188', '189', '190', '192', '193', '194', '195', '196', '197', '198', '200', '201', '202', '203', '205', '206', '208', '209', '211', '212', '213', '214', '215', '216', '217', '218', '219', '221', '222', '223', '224', '225', '226', '227', '228', '231', '232', '233', '235', '236', '238', '239', '240', '243', '244', '245', '246', '248', '249', '250', '253', '254', '255', '256', '257', '258', '260', '261', '263', '264', '265', '266', '267', '268', '270', '272', '273', '274', '275', '276', '277', '278', '280', '281', '282', '283', '284', '285', '287', '289', '291', '292', '293', '294', '296', '297', '298', '299', '300', '303', '304', '306', '307', '308', '309', '310', '311', '313', '314', '315', '316', '317', '318', '319', '320', '321', '323']
C.rare_files_to_class_ids = {'2': [1, 2], '3': [1, 2], '4': [1, 2], '5': [1, 2], '6': [1, 2], '7': [1, 2], '8': [1, 2], '9': [1, 2], '10': [1, 2], '12': [1, 2, 3], '13': [1, 2, 3], '14': [1, 2, 3, 4], '15': [1, 2, 3, 4], '18': [1, 2, 3, 4, 5], '19': [1, 2, 3, 4, 5], '20': [1, 2, 3, 5], '21': [1, 2, 3, 5], '22': [1, 2, 3, 5], '24': [6], '26': [6], '29': [6, 7], '32': [6, 7], '33': [6, 7], '34': [6, 7], '35': [6, 7], '36': [6, 7], '37': [6, 7], '38': [6, 7], '39': [6, 7], '40': [6, 7], '41': [6, 7, 8], '42': [6, 7, 8], '43': [6, 7, 8], '44': [6, 7, 8], '45': [6, 7, 8], '46': [6, 7, 8], '47': [4, 6, 7, 8], '48': [4, 6, 7, 8], '50': [1, 2, 3, 4, 6, 7, 8], '53': [1, 2, 3, 4, 6, 7, 8, 9], '54': [1, 2, 3, 4, 6, 7, 8, 9], '55': [1, 2, 3, 4, 6, 7, 8, 9], '56': [1, 2, 3, 6, 7, 8], '58': [1, 2, 3, 6, 7, 8], '59': [1, 2, 3, 6, 7, 8], '60': [1, 2, 3, 6, 7, 8], '61': [1, 2, 3, 6, 7, 8], '64': [1, 5, 10], '65': [1, 10], '66': [1, 10], '67': [1, 10], '68': [1, 3, 10], '69': [1, 3, 10], '70': [1, 3, 10], '71': [3], '72': [3], '73': [3], '77': [10], '78': [10], '79': [1, 10], '81': [10], '82': [10], '83': [11], '84': [11], '86': [11], '87': [11], '88': [11], '89': [11, 12], '90': [11, 12], '91': [11, 12], '93': [1, 6, 10, 11, 12], '94': [1, 6, 10, 11, 12], '95': [1, 4, 6, 9, 10, 11, 12], '96': [1, 4, 6, 9, 10, 11, 12], '99': [1, 4, 6, 9, 10, 11, 12, 13], '100': [1, 4, 9, 10, 11, 12, 13], '102': [1, 2, 3], '104': [2], '105': [2], '106': [2], '107': [14], '108': [14], '110': [14], '111': [14], '112': [14], '113': [14], '114': [14], '116': [14], '117': [14], '120': [14, 15], '121': [14, 15], '123': [15], '124': [15], '126': [15], '127': [15], '128': [15], '130': [15], '131': [15], '132': [15], '133': [15], '135': [16], '137': [16], '139': [16], '140': [16, 17], '142': [16, 17], '144': [16, 17], '145': [16, 17], '146': [16, 17], '147': [16, 17], '148': [16, 17], '149': [18], '151': [18], '152': [16, 18], '153': [16, 18], '154': [16, 18], '155': [5, 16, 18], '156': [5, 16, 18], '157': [5, 16, 18], '158': [5, 16, 18], '159': [5, 16, 18], '161': [19], '163': [20], '165': [20], '166': [20], '167': [20], '168': [20], '169': [20], '171': [21], '173': [21], '175': [21], '176': [19, 21], '177': [19, 21], '179': [19, 20, 21], '180': [19, 20, 21], '181': [19, 20, 21], '182': [19, 20], '183': [19, 20], '184': [19, 20], '186': [18, 19], '187': [18, 19], '188': [17], '189': [17], '190': [17], '192': [17], '193': [17], '194': [17, 21], '195': [17, 21], '196': [17, 21], '197': [16, 18, 19, 20], '198': [16, 18, 19, 20], '200': [22], '201': [22], '202': [22], '203': [22], '205': [22], '206': [22, 23], '208': [22, 23], '209': [24], '211': [24], '212': [24], '213': [24], '214': [24], '215': [24], '216': [24], '217': [24], '218': [24], '219': [24], '221': [25], '222': [25], '223': [25], '224': [25], '225': [25], '226': [25], '227': [25], '228': [25], '231': [24, 25], '232': [24, 25], '233': [23, 24, 25], '235': [23, 24, 25], '236': [23, 24, 25], '238': [23, 24, 25], '239': [24, 25, 26], '240': [24, 25, 26], '243': [24, 26], '244': [24, 26], '245': [22, 23, 24, 26], '246': [22, 23, 24, 26], '248': [5, 22, 23, 24, 26], '249': [5, 22, 23, 24, 26], '250': [5, 22, 23, 24, 26], '253': [4, 5, 22, 23, 24, 26], '254': [4, 5, 13, 22, 23, 24, 26, 27], '255': [4, 5, 13, 22, 23, 24, 26, 27], '256': [4, 5, 13, 22, 23, 24, 26, 27], '257': [4, 5, 13, 22, 23, 24, 26, 27], '258': [4, 5, 13, 22, 23, 24, 26, 27], '260': [13, 22, 23, 27], '261': [13, 22, 23, 27], '263': [13, 27], '264': [13, 27], '265': [13, 27, 28], '266': [13, 27, 28], '267': [13, 27, 28], '268': [13, 27, 28, 29], '270': [13, 27, 28, 29], '272': [27, 28, 29], '273': [27, 28, 29], '274': [27, 28, 29], '275': [28, 29], '276': [27, 28, 29], '277': [27, 28], '278': [27, 28], '280': [27, 29], '281': [27, 29], '282': [27, 29], '283': [27, 29], '284': [27, 29], '285': [27, 29], '287': [27, 28, 29], '289': [14, 15, 24, 27, 28], '291': [14, 15, 24, 27, 28], '292': [14, 15, 24, 27, 28], '293': [14, 15, 24, 27, 28], '294': [14, 15, 24, 27, 28], '296': [27, 30], '297': [27, 30], '298': [27, 30], '299': [27, 30], '300': [27, 30], '303': [27, 30], '304': [27, 31], '306': [27, 31], '307': [27, 31], '308': [27, 31], '309': [27, 31], '310': [27, 30, 31], '311': [27, 30, 31], '313': [30], '314': [30], '315': [30], '316': [30, 31], '317': [30, 31], '318': [30, 31], '319': [30, 31], '320': [30, 31], '321': [30, 31], '323': [30, 31]}
C.rare_files_multiplication = [
    max([C.rare_class_ids_factor[C.rare_class_ids.index(cid)] for cid in C.rare_files_to_class_ids[str(file_id)]])
    for file_id in C.rare_files_ids
]

with open(C.train_source) as f:
    train_files = [l.strip() for l in f if l.strip()]  # Just IDs

if getattr(C, 'rare_class_multiply', False):
    # Start with originals
    expanded_train_files = list(train_files)
    file_counts = {fname: 1 for fname in train_files}  # Track how many times we've added each

    # Oversample rare files: add (factor - 1) more (so total == factor)
    for idx, file_id in enumerate(C.rare_files_ids):
        fname = str(file_id)
        factor = C.rare_files_multiplication[idx]
        if fname in file_counts:
            add_count = max(factor - 1, 0)  # Never remove original
            expanded_train_files.extend([fname] * add_count)
            file_counts[fname] += add_count

    total_train_images = len(expanded_train_files)
else:
    expanded_train_files = list(train_files)
    total_train_images = len(train_files)
# C.manual_class_weights = None # Use None if not using manual weights

# <<< --- End Loss Function Configuration --- >>>
#
#   "sigmoid_gating"     – assumes Modality 0 (e.g., DIM) fused into base_feat via use_intensity_enhancement.
#                          For each *remaining* modality m (m>=1):
#                          - Learns *independent* per-pixel sigmoid weights g_m (range 0-1)
#                            based on [base_feat; aligned_feat_m].
#                          - Optionally transforms aligned_feat_m -> transformed_feat_m.
#                          Adds gated contributions to the base feature:
#                          fused = base_feat + Σ_{m>=1} g_m * transformed_feat_m
#

if C.use_intensity_enhancement:
    C.dataset_name += "-I"  # Append "I" if intensity merge
if C.sgate_fusion_mode == 'avg':
    C.dataset_name += "-AVG"  # Append "I" if intensity merge
if C.sgate_fusion_mode == 'add':
    C.dataset_name += "-ADD"  # Append "I" if intensity merge

C.dataset_name = (
    f"{C.dataset_name}"
    f"-{C.loss_function_name.upper()}"
    f"-{C.alignment_method.upper()}"
    f"-{C.fusion_method.upper()}"
    f"-{osp.basename(C.rgb_root_folder)}"
    f"-{C.modalities_tag}"
)

"""Train Config"""
C.lr = 1e-3 #6e-4 (better); 6e-5 #Reduced for training on top (3e-5)
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 10
C.nepochs = 500 # 1000
# C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
# Now set training iteration parameters according to rare classes if enabled
C.num_train_imgs = total_train_images
C.niters_per_epoch = C.num_train_imgs // C.batch_size + int(C.num_train_imgs % C.batch_size > 0)
C.file_len = total_train_images  # use for MM5Dataset or DataLoader, as appropriate


C.num_workers = 16
C.train_scale_array = [0.7, 0.85, 1, 1.2, 1.3, 1.5] # less zoom
# C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 250

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('./logs/log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    # Add new argument for model checkpoint
    parser.add_argument(
        '-mc', '--modelcheck', type=str, default=None,
        help='Path to the model checkpoint file')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
    if args.modelcheck:
        checkpoint = torch.load(C.pretrained_model)
        try:
            checkpoint = torch.load(C.pretrained_model)
            print(checkpoint.keys())
        except pickle.UnpicklingError as e:
            print("UnpicklingError: There was an error unpickling the file. This could indicate the file is corrupted.")
            print(e)
        except Exception as e:
            print("Error: An unexpected error occurred while loading the model.")
            print(e)
        
    # Assuming your model class is EncoderDecoder and it's defined in builder.py
    from models.builder import EncoderDecoder
    from config import config  # Make sure to import your config
    print("model:")
    # Instantiate your model (update parameters as necessary)
    # model = EncoderDecoder(cfg=config)
    
    # # Print model's state_dict keys
    # for key in model.state_dict().keys():
    #     print(key)
        