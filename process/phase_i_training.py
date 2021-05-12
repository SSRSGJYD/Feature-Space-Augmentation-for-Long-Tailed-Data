import argparse
import os

from datasets import get_dataset
from utils.general_util import get_device, load_config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='cifar10-LT_resnet18',
                    help='Which config is loaded from configs/phase_i')
parser.add_argument('-d', '--device', type=int, default=None,
                    help='Which gpu_id to use. If None, use cpu')
args = parser.parse_args()


def phase_i_train():
    pass


if __name__ == '__main__':
    config_file = os.path.join('configs', 'phase_i', args.config + '.yaml')
    config = load_config(config_file)

    device = get_device(args.device)
    dataset = get_dataset(config['dataset']['name'], **config['dataset']['kwargs'])
