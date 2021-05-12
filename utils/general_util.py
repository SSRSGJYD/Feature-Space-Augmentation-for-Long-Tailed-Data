import datetime
import logging
import logging.handlers
import os
import sys

import colorlog
import torch
import yaml


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
    'epoch': 1
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
    lr_scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    return lr_scheduler_cls(optimizer, **kwargs)


def create_criterion(name: str):
    if name == 'SoftmaxCrossEntropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise AttributeError('loss type %s is not recognized' % name)


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
