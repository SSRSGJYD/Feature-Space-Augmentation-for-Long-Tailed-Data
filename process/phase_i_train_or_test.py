import argparse
import os

import torch.utils.data

from datasets import get_dataset
from models import get_model
from utils.general_util import create_criterion, create_lr_scheduler, create_optimizer, default_train_state, get_device, \
    get_logger, load_config, load_state_dict_from_checkpoint, save_config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='cifar10-LT_resnet18',
                    help='Which config is loaded from configs/phase_i')
parser.add_argument('-d', '--device', type=int, default=None,
                    help='Which gpu_id to use. If None, use cpu')
parser.add_argument('-n', '--note', type=str, default='default_setting',
                    help='Note to identify this experiment, like "first_version"... Should not contain space')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='If set, log debug level, else info level')
args = parser.parse_args()


def main():
    config_file = os.path.join('configs', 'phase_i', args.config + '.yaml')
    config = load_config(config_file)
    path_prefix = os.path.join('phase_i', args.config, args.note)  # for logging, tensorboard, config_backup, etc.

    # backup config
    save_config(os.path.join(config['log']['path'], path_prefix, 'config_back.yaml'), config)
    # create logger to file and console
    logging_path = os.path.join(config['log']['path'], path_prefix, 'logging')
    logger = get_logger(name='phase_i' + args.config, logging_folder=logging_path, verbose=args.verbose)

    device = get_device(args.device)
    test_dataset = get_dataset(config['dataset']['name'], train=False, **config['dataset']['kwargs'])
    test_loader = torch.utils.data.DataLoader(test_dataset, config['test']['batch_size'], shuffle=False)

    model = get_model(config['model']['name'], num_classes=test_dataset.NUM_CLASSES, **config['model']['kwargs'])
    if config['checkpoint']['load_checkpoint'] is not None:
        load_checkpoint_file = os.path.join(
            config['checkpoint']['path'], path_prefix, config['checkpoint']['load_checkpoint'])
        model_state, train_state = load_state_dict_from_checkpoint(load_checkpoint_file)
        model.load_state_dict(model_state)
    else:
        train_state = default_train_state
    model = model.to(device)

    if config['phase'] == 'train':
        train_dataset = get_dataset(config['dataset']['name'], train=True, **config['dataset']['kwargs'])
        train_loader = torch.utils.data.DataLoader(train_dataset, config['train']['batch_size'], shuffle=True)
        phase_i_train(train_loader, test_loader, model, device, train_state, config, logger)
    else:
        phase_i_test(test_loader, model, device, config, logger)


def phase_i_train(train_loader, test_loader, model, device, train_state, config, logger):
    trained_parameters = model.parameters()
    optimizer = create_optimizer(config['train']['optimizer']['name'], trained_parameters,
                                 **config['train']['optimizer']['kwargs'])
    lr_scheduler = create_lr_scheduler(config['train']['lr_scheduler']['name'], optimizer,
                                       **config['train']['lr_scheduler']['kwargs'])
    criterion = create_criterion(config['train']['loss'])
    # actually train+test for every epoch
    for i_epoch in range(train_state['epoch'], config['train']['num_epoch'] + 1):
        logger.info('Begin to train epoch %d/%d...' % (i_epoch, config['train']['num_epoch']))
        for i_batch, (img, label) in enumerate(train_loader):
            img: torch.FloatTensor = img.to(device)
            label: torch.IntTensor = label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            prediction = torch.argmax(outputs, 1)
            loss.backward()
            if config['train']['gradient_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(trained_parameters, config['train']['gradient_clip'])
            optimizer.step()
            if (i_batch + 1) % config['log']['log_interval'] == 0:
                pass
        lr_scheduler.step()
        phase_i_test(test_loader, model, device, config, logger)


def phase_i_test(test_loader, model, device, config, logger):
    criterion = create_criterion(config['train']['loss'])

    logger.info('Begin to test...')
    for img, label in test_loader:
        img: torch.FloatTensor = img.to(device)
        label: torch.IntTensor = label.to(device)
        outputs = model(img)
        prediction = torch.argmax(outputs, 1)
        loss = criterion(outputs, label)


if __name__ == '__main__':
    main()
