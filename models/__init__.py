import torch

from models.resnet import *

all_models = {
    'resnet18':  resnet18,
    'resnet34':  resnet34,
    'resnet50':  resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


def get_model(name: str, num_classes: int, **kwargs) -> torch.nn.Module:
    assert name in all_models, 'model %s does not exist' % name
    return all_models[name](num_classes=num_classes, **kwargs)
