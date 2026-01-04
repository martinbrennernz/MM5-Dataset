import os.path as osp
import os
import sys
import time
import argparse
import importlib
from tqdm import tqdm
import numpy as np # Needed for class weight calculation

import torch
import torch.nn as nn # Use nn directly
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

# # Configuration
# from config import config
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

# Loss Functions
from models.losses import MultiClassDiceLoss, MultiClassFocalLoss # Adjust path if needed
from torch.nn import CrossEntropyLoss # Standard PyTorch CE Loss

# Dataloader
from dataloader.dataloader import get_train_loader

# Model Builder
from models.builder import EncoderDecoder as segmodel

# Utilities
from utils.init_func import init_weight, group_weight # Ensure these are correctly defined/imported
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, ensure_dir

# Tensorboard
from tensorboardX import SummaryWriter



# Now build the main parser
parser = argparse.ArgumentParser()
logger = get_logger()
# logger.setLevel(logging.DEBUG) # Uncomment for more verbose logging

os.environ['MASTER_PORT'] = config.get('master_port', '16971') # Use port from config or default

torch.cuda.empty_cache()

with Engine(custom_parser=parser) as engine:
    args = engine.args # Get parsed args from engine

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(engine.local_rank if engine.distributed else 0)}")
    else:
        logger.info("Using CPU")

    # --- Data Loader ---
    train_loader, train_sampler = get_train_loader(engine)
    logger.info(f"Dataloader created. Niters per epoch: {config.niters_per_epoch}")

    # --- TensorBoard Setup ---
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_log_dir = config.tb_dir
        ensure_dir(tb_log_dir)
        tb = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard writer created at: {tb_log_dir}")
    else:
        tb = None

    # --- Calculate Class Weights (if needed) ---
    class_weights = None
    if config.loss_class_weights_type == 'manual':
        if config.loss_manual_class_weights and len(config.loss_manual_class_weights) == config.num_classes:
            logger.info("Using manual class weights.")
            class_weights = torch.tensor(config.loss_manual_class_weights, dtype=torch.float)
        else:
            logger.warning("Manual class weights requested but not provided correctly in config. Using no weights.")
            config.loss_class_weights_type = 'none' # Fallback
    elif config.loss_class_weights_type == 'inverse_freq':
        logger.warning("Inverse frequency weights requested but calculation not implemented yet. Using no weights.")
        config.loss_class_weights_type = 'none' # Fallback for now

    # Move weights to device if calculated
    current_device = torch.device(f"cuda:{engine.local_rank}" if engine.distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
    if class_weights is not None:
         class_weights = class_weights.to(current_device)
         logger.info(f"Class weights moved to device: {current_device}")


    # --- Define Criterion(s) for Manual Calculation ---
    criterion = None # Main criterion for single loss cases
    combined_criteria = {} # For CEDice
    loss_name = config.loss_function_name.lower()
    ignore_label = config.background

    logger.info(f"Configuring loss function: {loss_name.upper()}")

    if loss_name == 'ce':
        criterion = CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label)
    elif loss_name == 'dice':
        # <<< --- Start of Fix --- >>>
        criterion = MultiClassDiceLoss(
            smooth=config.loss_params.get('dice_smooth', 1.0),
            weight=class_weights,
            average=config.loss_params.get('dice_average', 'macro'),
            average_foreground=config.loss_params.get('dice_average_foreground', True), 
            ignore_index=ignore_label,
            apply_softmax=True # Assumes model outputs logits
        )
        # <<< --- End of Fix --- >>>
    elif loss_name == 'focal':
        criterion = MultiClassFocalLoss(
            alpha=class_weights,
            gamma=config.loss_params.get('focal_gamma', 2.0),
            ignore_index=ignore_label,
            reduction='mean'
        )
    elif loss_name == 'cedice':
        # Create both criteria for manual combination
        combined_criteria['ce'] = CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label)
        # <<< --- Start of Fix --- >>>
        combined_criteria['dice'] = MultiClassDiceLoss(
            smooth=config.loss_params.get('dice_smooth', 1.0),
            weight=class_weights, # Use same weights? Or different?
            average=config.loss_params.get('dice_average', 'macro'),
            average_foreground=config.loss_params.get('dice_average_foreground', True), # Added this line
            ignore_index=ignore_label,
            apply_softmax=True
        )
        # <<< --- End of Fix --- >>>
        logger.info("Using combined CE + Dice loss (calculated manually).")
    else:
        logger.warning(f"Unsupported loss function '{loss_name}' in config. Using default CrossEntropyLoss.")
        loss_name = 'ce' # Fallback to CE
        criterion = CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label)


    # --- Network Setup ---
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # Instantiate model - Pass criterion=None, loss is calculated manually below
    model = segmodel(cfg=config, criterion=None, norm_layer=BatchNorm2d)
    logger.info("Model created.")

    # --- Optimizer Setup ---
    base_lr = config.lr
    params_list = group_weight([], model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not implemented.")
    logger.info(f"Optimizer {config.optimizer} created.")

    # --- LR Scheduler ---
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    logger.info("LR Scheduler created.")

    # --- Distributed Training Setup ---
    if engine.distributed:
        logger.info('Setting up distributed training...')
        if torch.cuda.is_available():
            model.cuda(engine.local_rank)
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank,
                                            find_unused_parameters=config.get('find_unused_parameters', False))
            logger.info(f'Model wrapped in DistributedDataParallel on rank {engine.local_rank}.')
    else:
        if torch.cuda.is_available():
            model.to(current_device)
            logger.info(f'Model moved to device: {current_device}')
        else:
             logger.info('Model running on CPU.')


    # --- Register state with Engine and restore checkpoint if provided ---
    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        logger.info(f"Restoring checkpoint from: {engine.continue_state_object}")
        engine.restore_checkpoint()
    else:
        logger.info("Starting training from scratch (or pretrained weights if specified in config).")
        if config.pretrained_model and not engine.continue_state_object:
             logger.info(f"Loading pretrained weights from: {config.pretrained_model}")
             model_to_load = model.module if engine.distributed else model
             if hasattr(model_to_load, 'init_weights'):
                 model_to_load.init_weights(pretrained=config.pretrained_model)
             else:
                 try:
                     from models.dual_segformer import load_dualpath_model # Adjust import
                     load_dualpath_model(model_to_load, config.pretrained_model)
                 except ImportError:
                      logger.warning("Cannot load pretrained weights: init_weights method or load_dualpath_model not found.")


    # --- Training Loop ---
    logger.info('Begin training...')
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format,
                    desc=f"Epoch {epoch}/{config.nepochs}")
        dataloader_iter = iter(train_loader)

        sum_loss = 0.0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            # --- Get Data ---
            try:
                minibatch = next(dataloader_iter)
            except StopIteration:
                logger.warning("Dataloader iterator exhausted unexpectedly. Resetting.")
                dataloader_iter = iter(train_loader)
                minibatch = next(dataloader_iter)

            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch.get('modal_x', None)
            
            logger.debug(f"Before forward: imgs={imgs.shape}, modal_xs={modal_xs.shape}, gts={gts.shape}")


            imgs = imgs.to(current_device, non_blocking=True)
            gts = gts.to(current_device, non_blocking=True)
            if modal_xs is not None:
                 modal_xs = modal_xs.to(current_device, non_blocking=True)

            # --- Forward Pass - Get Logits ---
            optimizer.zero_grad()
            try:
                model_to_call = model.module if engine.distributed else model
                if modal_xs is not None:
                     output = model_to_call.encode_decode(imgs, modal_xs)
                else:
                     output = model_to_call.encode_decode(imgs, None)

                if isinstance(output, tuple) and len(output) == 2:
                    logits, aux_logits = output
                else:
                    logits = output
                    aux_logits = None
                    
                    

            except Exception as e:
                 logger.error(f"Error during model forward pass at epoch {epoch}, iter {idx}: {e}", exc_info=True)
                 if modal_xs is not None: logger.error(f"Input shapes: RGB={imgs.shape}, ModalX={modal_xs.shape}, GT={gts.shape}")
                 else: logger.error(f"Input shapes: RGB={imgs.shape}, ModalX=None, GT={gts.shape}")
                 continue

            # --- Calculate Loss Manually ---
            loss = torch.tensor(0.0, device=current_device)
            aux_loss = torch.tensor(0.0, device=current_device)
            loss_ce_val = 0.0
            loss_dice_val = 0.0

            try:
                if loss_name == 'cedice':
                    loss_ce = combined_criteria['ce'](logits, gts.long())
                    loss_dice = combined_criteria['dice'](logits, gts.long())
                    loss = (config.loss_params.get('cedice_ce_weight', 0.5) * loss_ce +
                            config.loss_params.get('cedice_dice_weight', 0.5) * loss_dice)
                    loss_ce_val = loss_ce.item()
                    loss_dice_val = loss_dice.item()

                    if aux_logits is not None and config.get('aux_rate', 0) > 0:
                        aux_loss_ce = combined_criteria['ce'](aux_logits, gts.long())
                        aux_loss_dice = combined_criteria['dice'](aux_logits, gts.long())
                        aux_loss = (config.loss_params.get('cedice_ce_weight', 0.5) * aux_loss_ce +
                                    config.loss_params.get('cedice_dice_weight', 0.5) * aux_loss_dice)
                else:
                    loss = criterion(logits, gts.long())
                    if aux_logits is not None and config.get('aux_rate', 0) > 0:
                        aux_loss = criterion(aux_logits, gts.long())

            except Exception as e:
                 logger.error(f"Error calculating {loss_name.upper()} loss at epoch {epoch}, iter {idx}: {e}", exc_info=True)
                 logger.error(f"Logits shape: {logits.shape}, GT shape: {gts.shape}")
                 continue

            total_loss = loss + config.get('aux_rate', 0.4) * aux_loss

            # --- Backpropagation ---
            reduced_total_loss = total_loss.detach().clone()
            if engine.distributed:
                dist.all_reduce(reduced_total_loss, op=dist.ReduceOp.AVG)

            if not torch.isfinite(total_loss):
                 logger.warning(f"NaN or Inf loss detected (Rank {engine.local_rank if engine.distributed else 0}) at epoch {epoch}, iter {idx} BEFORE backward. Loss: {total_loss.item()}. Skipping backward.")
                 continue

            total_loss.backward()
            optimizer.step()

            # --- Logging ---
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += reduced_total_loss.item()
            average_loss_so_far = sum_loss / (idx + 1)

            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f' % total_loss.item() \
                        + ' total_loss=%.4f' % average_loss_so_far

            pbar.set_description(print_str, refresh=False)

            del loss, aux_loss, total_loss, reduced_total_loss

        # --- End of Epoch ---
        avg_loss_epoch = sum_loss / config.niters_per_epoch
        #logger.info(f"Epoch {epoch} finished. Average Loss: {avg_loss_epoch:.4f}")

        if tb is not None:
            tb.add_scalar('train/epoch_loss', avg_loss_epoch, epoch)
            tb.add_scalar('train/epoch_lr', lr, epoch)

        # --- Save Checkpoint ---
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                logger.info(f"Saving checkpoint for epoch {epoch}...")
                ensure_dir(config.checkpoint_dir)
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
                logger.info(f"Checkpoint saved to {config.checkpoint_dir}")

    # --- End of Training ---
    logger.info("Training finished.")
    if tb is not None:
        tb.close()

