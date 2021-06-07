import torch

from models.resnet import *
from models.resnet_cifar import *

all_models = {
    'resnet10':  resnet10,
    'resnet18':  resnet18,
    'resnet34':  resnet34,
    'resnet50':  resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet1202': resnet1202
}


def get_model(name: str, num_classes: int, **kwargs) -> torch.nn.Module:
    assert name in all_models, 'model %s does not exist' % name
    return all_models[name](num_classes=num_classes, **kwargs)
