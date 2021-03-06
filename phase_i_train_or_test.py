import argparse
import os

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataset
from models import get_model
from utils.general_util import create_criterion, create_lr_scheduler, create_optimizer, default_train_state, get_device, \
    get_logger, load_config, load_state_dict_from_checkpoint, save_config, save_state_dict_to_checkpoint, \
    set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='example',
                    help='Which config is loaded from configs/phase_i')
parser.add_argument('-d', '--device', type=int, default=None,
                    help='Which gpu_id to use. If None, use cpu')
parser.add_argument('-w', '--workers', type=int, default=0,
                    help='Number of workers in data loader')
parser.add_argument('-n', '--note', type=str, default='default_setting',
                    help='Note to identify this experiment, like "first_version"... Should not contain space')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='If set, log debug level, else info level')
args = parser.parse_args()


def main():
    config_file = os.path.join('configs', 'phase_i', args.config + '.yaml')
    config = load_config(config_file)
    if config['random_seed'] is not None:
        set_random_seed(config['random_seed'])
    path_prefix = os.path.join('phase_i', args.config, args.note)  # for logging, tensorboard, config_backup, etc.

    # backup config
    save_config(os.path.join(config['log']['path'], path_prefix, 'config_back.yaml'), config)
    # create logger to file and console
    logging_path = os.path.join(config['log']['path'], path_prefix, 'logging')
    logger = get_logger(name='phase_i' + args.config, logging_folder=logging_path, verbose=args.verbose)
    tensorboard_writer = SummaryWriter(os.path.join(config['tensorboard']['path'], path_prefix))
    checkpoints_folder = os.path.join(config['checkpoint']['path'], path_prefix)

    device = get_device(args.device)
    test_dataset = get_dataset(config['dataset']['name'], train=False, **config['dataset']['kwargs'])
    test_loader = torch.utils.data.DataLoader(test_dataset, config['test']['batch_size'], shuffle=False, num_workers=args.workers)

    model = get_model(config['model']['name'], num_classes=test_dataset.NUM_CLASSES, **config['model']['kwargs'])
    if config['checkpoint']['load_checkpoint'] is not None:
        load_checkpoint_file = os.path.join(checkpoints_folder, config['checkpoint']['load_checkpoint'])
        model_state, train_state = load_state_dict_from_checkpoint(load_checkpoint_file)
        model.load_state_dict(model_state)
    else:
        train_state = default_train_state
    model = model.to(device)

    if config['phase'] == 'train':
        train_dataset = get_dataset(config['dataset']['name'], train=True, **config['dataset']['kwargs'])
        train_loader = torch.utils.data.DataLoader(train_dataset, config['train']['batch_size'], shuffle=True, num_workers=args.workers)
        phase_i_train(train_loader, test_loader, model, device, train_state, config, logger, tensorboard_writer,
                      checkpoints_folder)
    else:
        phase_i_test(test_loader, model, device, config, logger)
    tensorboard_writer.close()


def phase_i_train(train_loader, test_loader, model, device, train_state, config, logger, tensorboard_writer,
                  checkpoint_folder):
    trained_parameters = model.parameters()
    optimizer = create_optimizer(config['train']['optimizer']['name'], trained_parameters,
                                 **config['train']['optimizer']['kwargs'])
    lr_scheduler = create_lr_scheduler(config['train']['lr_scheduler']['name'], optimizer,
                                       **config['train']['lr_scheduler']['kwargs'])
    criterion = create_criterion(config['train']['loss'])
    # actually train+test for every epoch
    best_acc = train_state['best_acc']
    for i_epoch in range(train_state['epoch'], config['train']['num_epoch'] + 1):
        logger.info('Begin to train epoch %d/%d...' % (i_epoch, config['train']['num_epoch']))
        total_samples, log_samples = 0, 0
        total_corrects, log_corrects = 0, 0
        total_loss, log_loss = 0.0, 0.0
        model.train()

        for i_batch, (img, label, _) in enumerate(train_loader):
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

            total_samples += len(img)
            log_samples += len(img)
            total_corrects += (prediction == label).type(torch.int32).sum().item()
            log_corrects += (prediction == label).type(torch.int32).sum().item()
            total_loss += len(img) * loss.item()
            log_loss += len(img) * loss.item()

            # only save first batch images to tensorboard
            # if i_batch == 0:
            #     tensorboard_writer.add_image('image/train', img[0].cpu(), i_epoch)
            if (i_batch + 1) % config['log']['log_interval'] == 0:
                log_loss = log_loss / log_samples
                log_acc = log_corrects / log_samples
                logger.info('finish train batch %d. loss: %1.4f. acc: %1.4f' % (i_batch + 1, log_loss, log_acc))
                log_samples = 0
                log_corrects = 0
                log_loss = 0.0
        lr_scheduler.step()

        epoch_loss = total_loss / total_samples
        epoch_acc = total_corrects / total_samples
        logger.info('finish train epoch %d/%d. loss: %1.4f. acc: %1.4f'
                    % (i_epoch, config['train']['num_epoch'], epoch_loss, epoch_acc))
        tensorboard_writer.add_scalar('loss/train', epoch_loss, i_epoch)
        tensorboard_writer.add_scalar('acc/train', epoch_acc, i_epoch)
        test_acc = phase_i_test(test_loader, model, device, config, logger, tensorboard_writer, i_epoch)
        train_state['epoch'] = i_epoch
        train_state['acc'] = test_acc
        if i_epoch % config['checkpoint']['save_checkpoint_interval'] == 0 or i_epoch == config['train']['num_epoch']:
            save_state_dict_to_checkpoint(os.path.join(checkpoint_folder, 'model_epoch_%04d.pt' % i_epoch),
                                          model.state_dict(), train_state)
        if test_acc > best_acc:
            best_acc = test_acc
            train_state['best_acc'] = best_acc
            train_state['best_acc_epoch'] = i_epoch
            save_state_dict_to_checkpoint(os.path.join(checkpoint_folder, 'best_model.pt'),
                                          model.state_dict(), train_state)


def phase_i_test(test_loader, model, device, config, logger, tensorboard_writer=None, i_epoch=None):
    criterion = create_criterion(config['test']['loss'])

    if i_epoch is None:
        logger.info('Begin to test...')
    else:
        logger.info('Begin to test epoch %d...' % i_epoch)
    model.eval()
    with torch.no_grad():
        total_samples = 0
        total_corrects = 0
        total_loss = 0.0
        for i_batch, (img, label, _) in enumerate(test_loader):
            img: torch.FloatTensor = img.to(device)
            label: torch.IntTensor = label.to(device)
            outputs = model(img)
            prediction = torch.argmax(outputs, 1)
            loss = criterion(outputs, label)

            # only save first batch images to tensorboard
            # if i_epoch is not None and i_batch == 0:
            #     tensorboard_writer.add_image('image/test', img[0].cpu(), i_epoch)

            total_samples += len(img)
            total_corrects += (prediction == label).type(torch.int32).sum().item()
            total_loss += len(img) * loss.item()
        epoch_loss = total_loss / total_samples
        epoch_acc = total_corrects / total_samples
        if i_epoch is None:
            logger.info('finish test. loss: %1.4f. acc: %1.4f' % (epoch_loss, epoch_acc))
        else:
            logger.info('finish epoch %d. loss: %1.4f. acc: %1.4f' % (i_epoch, epoch_loss, epoch_acc))
            tensorboard_writer.add_scalar('loss/test', epoch_loss, i_epoch)
            tensorboard_writer.add_scalar('acc/test', epoch_acc, i_epoch)
    return epoch_acc


if __name__ == '__main__':
    main()
