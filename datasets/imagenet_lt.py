import torch.utils.data


class ImageNetLTDataset(torch.utils.data.Dataset):
    NAME = 'ImageNet-LT'
    NUM_CLASSES = 1000
    IM = 256

    def __init__(self, train: bool, **kwargs):
        super().__init__()
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
