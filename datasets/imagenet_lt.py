import torch.utils.data
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import os

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class ImageNetLTDataset(torch.utils.data.Dataset):
    NAME = 'ImageNet-LT'
    NUM_CLASSES = 1000
    IM = 256

    def __init__(self, train: bool, **kwargs):
        super().__init__()
        # root path is the folder name containing imageNet_ILSVRC2012 dataset
        self.rootpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imageNet_ILSVRC2012")
        self.img_path = []
        self.labels = []
        self.train = train
        if ( train ) :
            txtpath = os.path.join(self.rootpath, "ImageNet_LT_train.txt")
        else:
            txtpath = os.path.join(self.rootpath, "ImageNet_LT_test.txt")
        self.class_samples = {}
        with open(txtpath) as f:
            for line in f:
                self.img_path.append(line.split()[0])
                self.labels.append(int(line.split()[1]))

        sum_samples = 0 # total number of samples in imageNet-LT 

        self.many_shot = set()
        self.medium_shot = set()
        self.few_shot = set()

        for c in range(self.NUM_CLASSES):
            self.class_samples[c] = [(l,os.path.basename(uuid).split('.')[0]) for l,uuid in zip(self.labels,self.img_path) if l == c]
            sum_samples += len(self.class_samples[c])
            if len(self.class_samples[c]) >100:
                self.many_shot.add(c)
            elif len(self.class_samples[c]) <= 100 and len(self.class_samples[c]) > 20:
                self.medium_shot.add(c)
            elif  len(self.class_samples[c]) <= 20:
                self.few_shot.add(c)

        # sort the class_samples in descending order
        self.class_samples = {k: v for k, v in sorted(self.class_samples.items(), key=lambda item: len(item[1]),reverse=True)}
        self.head_classes = set()
        head_samples = 0 # total number of smaples in head classes 
        for key,value in self.class_samples.items():
            head_samples += len(value)
            self.head_classes.add(key)
            if head_samples/sum_samples >= 0.9:# use h_r = 0.9 
                break

    def __getitem__(self, item):
        path = self.img_path[item]
        label = self.labels[item]
        
        with open(os.path.join(self.rootpath,path), 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if(self.train):
            sample = data_transforms["train"](sample)
        else:
            sample = data_transforms["test"](sample)
        return sample, label, os.path.basename(path).split('.')[0]

    def __len__(self):
        return len(self.labels)
