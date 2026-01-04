# encoding: utf-8
import os
import sys
import time
import random
import argparse
import logging
from collections import OrderedDict, defaultdict

import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist

class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            # self.log_fout.write(formatted)
            # self.log_fout.write('\n')
            # self.log_fout.flush()
            return formatted

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_warn(msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_err(msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_omitted(msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)

_default_level_name = os.getenv('ENGINE_LOGGING_LEVEL', 'INFO')
_default_level = logging.getLevelName(_default_level_name.upper())

def get_logger(log_dir=None, log_file=None, formatter=LogFormatter):
    logger = logging.getLogger()
    logger.setLevel(_default_level)
    del logger.handlers[:]

    if log_dir and log_file:
        ensure_dir(log_dir)
        LogFormatter.log_fout = True
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
    stream_handler.setLevel(0)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger()

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)

    return tensor


def load_restore_model(model, model_file):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        elif 'module' in state_dict.keys():
            state_dict = state_dict['module']
    else:
        state_dict = model_file
    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=True)
    
    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model

def load_model(model, model_file, is_restore=False, show_warning=True):
    """
    Load model weights from a file.

    Args:
        model (nn.Module): The model instance to load weights into.
        model_file (str): Path to the checkpoint file (.pth).
        is_restore (bool): If True, indicates restoring a training checkpoint
                           (might contain optimizer state, etc.). Defaults to False.
        show_warning (bool): If True, show warnings for missing/unexpected keys.

    Returns:
        nn.Module: The model with loaded weights.
    """
    if not model_file or not os.path.exists(model_file):
        logger.error(f"Model file not found or not specified: {model_file}")
        # Depending on use case, either raise error or return original model
        # raise FileNotFoundError(f"Model file not found: {model_file}")
        logger.warning("Returning model without loading weights.")
        return model

    t_start = time.time()
    try:
        # Load checkpoint onto CPU first to avoid GPU memory spike
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        logger.info(f"Successfully loaded checkpoint file: {model_file}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint file {model_file}: {e}", exc_info=True)
        # Raise error or return original model
        raise IOError(f"Failed to load checkpoint file: {model_file}") from e
        # return model

    t_ioend = time.time()

    # Check if the loaded state_dict is nested (e.g., under 'model' or 'state_dict')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # No else needed, assume state_dict is the model weights directly if not nested

    # --- Clean state_dict keys ---
    # Remove 'module.' prefix if it exists (from DistributedDataParallel saving)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # --- Load state_dict into model ---
    # <<< --- Start of Fix --- >>>
    # Load with strict=False to ignore keys like 'criterion.weight' during evaluation
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    # <<< --- End of Fix --- >>>

    # --- Log loading summary ---
    if show_warning:
        if missing_keys:
            logger.warning(f"Missing key(s) in state_dict: {', '.join(missing_keys)}")
        # Only log unexpected keys if they are truly unexpected (not criterion etc.)
        # Filter out common keys that might be intentionally ignored during eval
        keys_to_ignore_warning = ['criterion.weight'] # Add others if needed
        filtered_unexpected = [k for k in unexpected_keys if k not in keys_to_ignore_warning]
        if filtered_unexpected:
            logger.warning(f"Unexpected key(s) in state_dict: {', '.join(filtered_unexpected)}")
        elif unexpected_keys: # Log that criterion weights were ignored if they were present
             if any(k in keys_to_ignore_warning for k in unexpected_keys):
                  logger.info(f"Ignored expected keys during eval loading (e.g., criterion weights): {', '.join(k for k in unexpected_keys if k in keys_to_ignore_warning)}")


    del state_dict, new_state_dict # Free memory
    t_end = time.time()
    logger.info(f"Loaded model weights from {model_file}. Time usage: IO: {t_ioend - t_start:.2f}s, Load: {t_end - t_ioend:.2f}s")

    return model

def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.system('rm -rf {}'.format(target))
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print('conflict !!!')


def _dbg_interactive(var, value):
    from IPython import embed
    embed()