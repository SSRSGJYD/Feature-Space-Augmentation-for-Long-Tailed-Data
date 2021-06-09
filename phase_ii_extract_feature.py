import argparse
import numpy as np
import os
import torch.utils.data
import torch.nn.functional as F
from datasets import get_dataset
from models import get_model
from utils.general_util import get_device, load_config, load_state_dict_from_checkpoint, set_random_seed
from utils.phase_ii_util import GradCam

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='example',
                    help='Which config is loaded from configs/phase_ii')
parser.add_argument('-d', '--device', type=int, default=None,
                    help='Which gpu_id to use. If None, use cpu')
parser.add_argument('-n', '--note', type=str, default='default_setting',
                    help='Note to identify this experiment, like "first_version"... Should not contain space')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='If set, log debug level, else info level')
args = parser.parse_args()


def main():
    config_file = os.path.join('configs', 'phase_ii', args.config + '.yaml')
    config = load_config(config_file)
    if config['random_seed'] is not None:
        set_random_seed(config['random_seed'])
    
    device = get_device(args.device)
    train_dataset = get_dataset(config['dataset']['name'], train=True, **config['dataset']['kwargs'])
    train_loader = torch.utils.data.DataLoader(train_dataset, config['test']['batch_size'], shuffle=False)
        
    model = get_model(config['model']['name'], num_classes=train_dataset.NUM_CLASSES, **config['model']['kwargs'])
    assert config['checkpoint']['load_checkpoint'] is not None
    model_state, train_state = load_state_dict_from_checkpoint(config['checkpoint']['load_checkpoint'])
    model.load_state_dict(model_state)
    model = model.to(device)
    gradcam = GradCam(model, config['feature']['cam_layers'].split(','))
    phase_ii_test(train_loader, model, gradcam, device, config)


def phase_ii_test(test_loader, model, gradcam, device, config):
    model.eval()
    scores_per_class = torch.zeros((test_loader.dataset.NUM_CLASSES, test_loader.dataset.NUM_CLASSES))
    path_prefix = os.path.join(args.config, args.note)
    
    for i_batch, (img, label, uuids) in enumerate(test_loader):
        img: torch.FloatTensor = img.to(device)
        label: torch.IntTensor = label.to(device)
        outputs = model(img)
        scores = F.softmax(outputs, dim=1).cpu().detach()
        last = label[0]
        gradcam.cat_info()
        change_indices = [0]
        for i in range(1, len(label)):
            if label[i] != last:
                change_indices.append(i)
                last = label[i]

        change_indices.append(len(label))
        for i in range(len(change_indices)-1):
            start = change_indices[i]
            end = change_indices[i+1]
            c = int(label[start])
            for j in range(start, end):
                model.zero_grad()
                gradcam.cal_grad(outputs[j:j+1], c)
                result = gradcam.cal_cam(j)
                for k, cam in result.items():
                    save_folder = os.path.join(config['feature']['path'], path_prefix, k)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    # save record
                    np.savez(os.path.join(save_folder, '{}.npz'.format(uuids[j])), 
                        cam=cam.numpy(),
                        feature=gradcam.vis_info[k]['output'][j].cpu().numpy())
                # update scores
                scores_per_class[c].add_(scores[j])
        gradcam.reset_info()

    # average scores per class
    for c in range(test_loader.dataset.NUM_CLASSES):
        scores_per_class[c].div_(len(test_loader.dataset.class_samples[c]))
    torch.save(scores_per_class, os.path.join(config['feature']['path'], path_prefix, 'scores_per_class.pt'))
    

if __name__ == '__main__':
    main()
