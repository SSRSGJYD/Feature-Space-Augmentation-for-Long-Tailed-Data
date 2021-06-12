import numpy as np
import torch.utils.data
from torchvision import transforms


class ExampleDataset(torch.utils.data.Dataset):
    NAME = 'example'

    def __init__(self, train: bool, im: int, **kwargs):
        super().__init__()
        self.IM = im
        self.NUM_CLASSES = 2
        self.head_classes = set([0])
        if train:
            self.class_samples = {
                0:[(0, uuid) for uuid in range(im * 20)], 
                1:[(1, uuid) for uuid in range(im * 20, im*20 + 20)]
            }
        else:
            self.class_samples = {
                0:[(0, uuid) for uuid in range(im * 10)], 
                1:[(1, uuid) for uuid in range(im * 10, im*10 + 10)]
            }
        self.samples = []
        for c, l in self.class_samples.items():
            self.samples.extend(l)
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        c, uuid = self.samples[item]
        image = c * 255 * np.ones((224, 224, 3), dtype=np.float)
        tensor = self.transform(image).float()
        return tensor, c, uuid

    def __len__(self):
        return len(self.samples)

    def load_sample(self, sample):
        c, uuid = sample
        image = c * 255 * np.ones((224, 224, 3), dtype=np.float)
        tensor = self.transform(image).float()
        return tensor, c, uuid