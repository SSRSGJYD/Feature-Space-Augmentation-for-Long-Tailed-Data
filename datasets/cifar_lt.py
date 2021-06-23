import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Cifar10LTDataset(torchvision.datasets.CIFAR10):
    NAME = 'cifar10-LT'
    NUM_CLASSES = 10
    IM = None  # 10, 20, 50, 100 or 200

    def __init__(self, train: bool, im: int, **kwargs):
        super().__init__(root='./data', train=train, transform=None, target_transform=None, download=True)
        self.NUM_CLASSES = 10
        self.IM = 1.0 / im
        self.train = train
        self.head_classes = set([0, 1, 2, 3, 4, 5, 6])
        if train:
            img_num_list = self.get_img_num_per_cls()
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.labels = self.targets
        
        self.class_samples = dict()
        for i in range(len(self.labels)):
            if self.class_samples.get(self.labels[i]) is None:
                self.class_samples[self.labels[i]] = []
            self.class_samples[self.labels[i]].append((self.labels[i], i))

    def __getitem__(self, item):
        img, label = self.data[item], self.labels[item]
        img = Image.fromarray(img)
        img = self.transform(img).float()
        return img, label, item

    def __len__(self):
        return len(self.labels)

    def get_img_num_per_cls(self):
        img_max = len(self.data) / self.NUM_CLASSES
        img_num_per_cls = []
        for cls_idx in range(self.NUM_CLASSES):
            num = img_max * (self.IM ** (cls_idx / (self.NUM_CLASSES - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets


class Cifar100LTDataset(torchvision.datasets.CIFAR100):
    NAME = 'Cifar100-LT'
    NUM_CLASSES = 100
    IM = None  # 10, 20, 50, 100 or 200

    def __init__(self, train: bool, im: int, **kwargs):
        super().__init__(root='/data3/zhouchenyu/DataSets', train=train, transform=None, target_transform=None, download=False)
        self.IM = im
        self.train = train
        self.head_classes = set([0])

        if train:
            img_num_list = self.get_img_num_per_cls()
            print('img_num_list:', img_num_list)
            total_sum = sum(img_num_list)
            temp_sum = 0
            for i in range(100):
                temp_sum += img_num_list[i]
                if temp_sum > total_sum * 0.9:
                    break
                self.head_classes.add(i)
            print('head_classes:', self.head_classes)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])


        self.labels = self.targets
        self.class_samples = dict()
        for i in range(len(self.labels)):
            if self.class_samples.get(self.labels[i]) is None:
                self.class_samples[self.labels[i]] = []
            self.class_samples[self.labels[i]].append((self.labels[i], i))

    def __getitem__(self, item):
        img, label = self.data[item], self.labels[item]
        img = Image.fromarray(img)
        img = self.transform(img).float()
        return img, label, item

    def __len__(self):
        return len(self.labels)

    def get_img_num_per_cls(self):
        img_max = len(self.data) / self.NUM_CLASSES
        img_num_per_cls = []
        for cls_idx in range(self.NUM_CLASSES):
            num = img_max * ((1.0 / self.IM) ** (cls_idx / (self.NUM_CLASSES - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def load_sample(self, sample):
        c, uuid = sample
        image = Image.fromarray(self.data[uuid])
        tensor = self.transform(image).float()
        return tensor, c, uuid
