import torch.utils.data


class Cifar10LTDataset(torch.utils.data.Dataset):
    NAME = 'cifar10-LT'
    NUM_CLASSES = 10
    IM = None  # 10, 20, 50, 100 or 200

    def __init__(self, train: bool, im: int, **kwargs):
        super().__init__()
        # TODO
        pass

    def __getitem__(self, item):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass


class Cifar100LTDataset(torch.utils.data.Dataset):
    NAME = 'cifar100-LT'
    NUM_CLASSES = 100
    IM = None  # 10, 20, 50, 100 or 200

    def __init__(self, train: bool, im: int, **kwargs):
        super().__init__()
        # TODO
        pass

    def __getitem__(self, item):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass
