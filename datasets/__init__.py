import torch.utils.data

from datasets.example import ExampleDataset
from datasets.cifar_lt import Cifar100LTDataset, Cifar10LTDataset
from datasets.feature_dataset import FeatureDataset
from datasets.i_naturalist import INaturalist2017Dataset, INaturalist2018Dataset
from datasets.imagenet_lt import ImageNetLTDataset
from datasets.places_lt import PlacesLTDataset

all_datasets = {
    'example':          ExampleDataset,
    'Cifar10-LT':       Cifar10LTDataset,
    'cifar100-LT':      Cifar100LTDataset,
    'ImageNet-LT':      ImageNetLTDataset,
    'Places-LT':        PlacesLTDataset,
    'iNaturalist-2017': INaturalist2017Dataset,
    'iNaturalist-2018': INaturalist2018Dataset,

    'feature_dataset':  FeatureDataset
}


def get_dataset(name: str, train: bool, **kwargs) -> torch.utils.data.Dataset:
    assert name in all_datasets, 'dataset %s does not exist' % name
    return all_datasets[name](train=train, **kwargs)
