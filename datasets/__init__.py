import torch.utils.data

from datasets.example import ExampleDataset
from datasets.cifar_lt import Cifar100LTDataset, Cifar10LTDataset
from datasets.imagenet_lt import ImageNetLTDataset

all_datasets = {
    'example':          ExampleDataset,
    'Cifar10-LT':       Cifar10LTDataset,
    'Cifar100-LT':      Cifar100LTDataset,
    'ImageNet-LT':      ImageNetLTDataset,
}


def get_dataset(name: str, train: bool, **kwargs) -> torch.utils.data.Dataset:
    assert name in all_datasets, 'dataset %s does not exist' % name
    return all_datasets[name](train=train, **kwargs)
