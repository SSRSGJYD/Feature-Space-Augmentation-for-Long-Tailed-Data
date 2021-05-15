import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """This class is the dataset in stage 3."""

    def __init__(self, image_dataset, config):
        self.Nt = config['Nt']
        self.Na = config['Na']
        self.Nf = config['Nf']
        self.ts = config['ts']
        self.tg = config['tg']
        self.feature_folder = config['feature']['path']
        self.cam_layer = config['finetune']['cam_layer']

        self.NUM_CLASSES = image_dataset.NUM_CLASSES
        self.head_classes = image_dataset.head_classes
        self.class_samples = image_dataset.class_samples
        self.head_class_samples = []
        self.tail_class_samples = []
        for c, samples in self.class_samples:
            if c in self.head_classes:
                self.head_class_samples += samples
            else:
                self.tail_class_samples += samples

        scores_per_class = torch.load(os.path.join(config['feature']['path'], 'scores_per_class.pt'))
        for c in range(self.NUM_CLASSES):
            scores_per_class[c, c] = 0
        _, topk = torch.topk(scores_per_class, self.Nf, dim=0)
        self.confusing_head_classes = topk.t().tolist()

    def read_feature(self, sample, need_cam=True):
        record = np.load(os.path.join(self.feature_folder, self.cam_layer, '{}.npz'.format(sample[1])))
        feature = record['feature']
        cam = record['cam'] if need_cam else None
        return feature, cam

    def fusion_feature(self, tail_sample):
        # feature fusion
        tail_feature, tail_cam = self.read_feature(tail_sample)
        tail_feature = np.where(tail_cam > self.ts, tail_feature, 0)
        tail_features = [tail_feature]

        tail_class = tail_sample[0]
        confusing_classes = self.confusing_head_classes[tail_class]
        for head_class in confusing_classes:
            # sample one sample from each of the Na head classes
            confusing_sample = random.sample(self.class_samples[head_class], 1)     
            head_feature, head_cam = self.read_feature(confusing_sample)
            head_feature = np.where(head_cam < self.tg, head_feature, 0)
            combine_mask = np.random.rand(head_feature.shape[0], head_feature.shape[1])
            combine_mask = np.where(combine_mask > 0.5, 1, 0)
            fusion_feature = combine_mask * tail_feature + (1-combine_mask) * head_feature
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
            label[idx] = head_sample[0]
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
