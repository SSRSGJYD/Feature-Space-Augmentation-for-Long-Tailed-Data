import os

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
