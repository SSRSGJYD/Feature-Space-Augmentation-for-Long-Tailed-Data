import datetime
import logging
import logging.handlers
import math
import os
import random
import sys

from bisect import bisect_right
import colorlog
import numpy as np
import torch
import yaml

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def get_device(device: int = None) -> torch.device:
    if device is None:
        return torch.device('cpu')
    else:
        return torch.device('cuda:%d' % device)


def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_config(config_filename: str) -> dict:
    with open(config_filename, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config_filename: str, config: dict):
    maybe_create_path(os.path.dirname(config_filename))
    with open(config_filename, 'w', encoding='utf8') as f:
        yaml.safe_dump(config, f, sort_keys=False)


default_train_state = {
    'epoch':          1,
    'acc':            0.0,
    'best_acc':       0.0,
    'best_acc_epoch': 1
}


def load_state_dict_from_checkpoint(checkpoint_file: str) -> (dict, dict):
    assert os.path.exists(checkpoint_file), 'checkpoint file %s does not exist' % checkpoint_file
    state: dict = torch.load(checkpoint_file)
    model_state = state.pop('model_state')
    train_state = state
    return model_state, train_state


def save_state_dict_to_checkpoint(checkpoint_file: str, model_state: dict, train_state: dict = None):
    maybe_create_path(os.path.dirname(checkpoint_file))
    if train_state is None:
        train_state = default_train_state
    state = train_state.copy()
    state['model_state'] = model_state
    torch.save(state, checkpoint_file)


def create_optimizer(name: str, params, **kwargs) -> torch.optim.Optimizer:
    optimizer_cls = getattr(torch.optim, name)
    optimizer = optimizer_cls(params, **kwargs)
    return optimizer


def create_lr_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs):
    if name == 'warmup':
        lr_scheduler_cls = WarmupMultiStepLR
    else:
        lr_scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    return lr_scheduler_cls(optimizer, **kwargs)


def create_criterion(name: str):
    if name == 'SoftmaxCrossEntropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise AttributeError('loss type %s is not recognized' % name)


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name, logging_folder=None, verbose=False, logging_file_prefix=None):
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.getLogger('PIL').setLevel(logging.INFO)  # prevent PIL logging many debug msgs
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # prevent matplotlib logging many debug msgs

    # root logger to log everything
    root_logger = logging.root
    root_logger.setLevel(level)
    if not root_logger.handlers:
        format_str = '%(asctime)s [%(threadName)s] %(levelname)s [%(name)s] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG':    'cyan',
                  'INFO':     'green',
                  'WARNING':  'bold_yellow',
                  'ERROR':    'red',
                  'CRITICAL': 'bold_red', }
        color_formatter = colorlog.ColoredFormatter(cformat, date_format, log_colors=colors)
        plain_formatter = logging.Formatter(format_str, date_format)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(color_formatter)
        root_logger.addHandler(stream_handler)
        # Logging to file
        if logging_folder is not None:
            maybe_create_path(logging_folder)
            logging_filename = datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S') + '.log'
            if logging_file_prefix is not None:
                logging_filename = logging_file_prefix + '_' + logging_filename
            logging_filename = os.path.join(logging_folder, logging_filename)
            file_handler = logging.handlers.RotatingFileHandler(
                logging_filename, maxBytes=5 * 1024 * 1024, encoding='utf8')  # 5MB per file
            file_handler.setFormatter(plain_formatter)
            root_logger.addHandler(file_handler)
    return logger
