import torch.utils.data


class INaturalist2017Dataset(torch.utils.data.Dataset):
    NAME = 'iNaturalist-2017'
    NUM_CLASSES = 5089
    IM = None

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


class INaturalist2018Dataset(torch.utils.data.Dataset):
    NAME = 'iNaturalist-2018'
    NUM_CLASSES = 8142
    IM = None

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
