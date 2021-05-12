import torch.utils.data


class PlacesLTDataset(torch.utils.data.Dataset):
    NAME = 'Places-LT'
    NUM_CLASSES = 365
    IM = 996

    def __init__(self, train: bool, **kwargs):
        super().__init__()
        # TODO
        pass

    def __getitem__(self, item):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass
