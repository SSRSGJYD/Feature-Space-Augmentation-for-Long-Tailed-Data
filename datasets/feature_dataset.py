import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """This class is the dataset in stage 2 and 3."""

    def __init__(self, config):
        self.Nt = config.Nt
        self.Na = config.Na
        self.Nf = config.Nf
        self.ts = config.ts
        self.tg = config.tg
        self.gamma = config.gamma

        self.head_classes = json.load(config.head_classes)
        self.class_samples = json.load(config.class_samples)
        self.confusing_head_classes = json.load(config.confusing_head_classes)
        self.head_class_samples = []
        self.tail_class_samples = []
        for c, samples in self.class_samples:
            if c in self.head_classes:
                self.head_class_samples += samples
            else:
                self.tail_class_samples += samples

    def read_feature(self, sample, need_cam=True):
        feature = np.load('')
        cam = np.load('') if need_cam else None
        return feature, cam

    def fusion_feature(self, tail_sample):
        # feature fusion
        tail_feature, tail_cam = self.read_feature(tail_sample)
        tail_feature = np.where(tail_cam > self.ts, tail_feature, 0)
        tail_feature = np.mean(tail_feature, axis=(1,2))
        tail_features = [tail_feature]

        tail_class = tail_sample['class']
        confusing_classes = self.confusing_head_classes[tail_class]
        for head_class in confusing_classes:
            # sample one sample from each of the Na head classes
            confusing_sample = random.sample(self.class_samples[head_class], 1)     
            head_feature, head_cam = self.read_feature(confusing_sample)
            head_feature = np.where(head_cam < self.tg, head_feature, 0)
            head_feature = np.mean(head_feature, axis=(1,2))
            fusion_feature = self.gamma * tail_feature + (1-self.gamma) * head_feature
            tail_features.append(fusion_feature)
        return tail_features
 
    def __len__(self):
        return len(self.tail_class_samples)

    def __getitem__(self, idx):
        batch_features = self.fusion_feature(self.tail_class_samples[idx])
        label = torch.zeros(2*(1+self.Na), dtype=torch.long)

        # head class samples
        idx = 1 + self.Na
        head_samples = random.sample(self.head_class_samples, 1+self.Na)
        for head_sample in head_samples:
            head_feature, _ = self.read_feature(head_sample, False)
            batch_features.append(head_feature)
            label[idx] = head_sample['class']
            idx += 1
        
        return batch_features, label

    def collate_fn(self, batch):
        features = []
        labels = []
        for batch_features, label in batch:
            features += batch_features
            labels.append(label)

        input_tensor = torch.from_numpy(np.stack(features)).contiguous()
        label = torch.cat(labels)
        return input_tensor, label
