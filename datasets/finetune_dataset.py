import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class FinetuneDataset(Dataset):
    """This class is the dataset in stage 2 and 3."""

    def __init__(self, image_dataset, config):
        self.Nt = config['finetune']['Nt']
        self.Na = config['finetune']['Na']
        self.Nf = config['finetune']['Nf']
        self.ts = config['finetune']['ts']
        self.tg = config['finetune']['tg']
        self.cam_layer = config['feature']['cam_layers']

        self.image_dataset = image_dataset
        self.NUM_CLASSES = image_dataset.NUM_CLASSES
        self.head_classes = image_dataset.head_classes
        self.class_samples = image_dataset.class_samples
        self.head_class_samples = []
        self.tail_class_samples = []
        for c, samples in self.class_samples.items():
            if c in self.head_classes:
                self.head_class_samples.extend(samples)
            else:
                self.tail_class_samples.extend(samples)

        scores_per_class = torch.load(os.path.join(config['feature']['path'], 'scores_per_class.pt'))
        for c in range(self.NUM_CLASSES):
            scores_per_class[c, c] = 0
        _, topk = torch.topk(scores_per_class, self.Nf, dim=0)
        self.confusing_head_classes = topk.t().tolist()
 
    def __len__(self):
        return len(self.tail_class_samples)

    def __getitem__(self, idx):
        tail_sample = self.tail_class_samples[idx]
        tail_sample_img, tail_class, _ = self.image_dataset.load_sample(tail_sample)
        tail_images = [tail_sample_img]
        tail_labels = [tail_class]

        # confusing class samples
        confusing_classes = self.confusing_head_classes[tail_class]
        for head_class in confusing_classes:
            # sample one sample from each of the Na head classes
            confusing_sample = random.choice(self.class_samples[head_class])
            confusing_sample_img, c, _ = self.image_dataset.load_sample(confusing_sample)
            tail_images.append(confusing_sample_img)
            tail_labels.append(c)

        # head class samples
        head_images = []
        head_labels = []
        head_samples = random.sample(self.head_class_samples, 1+self.Na)
        for head_sample in head_samples:
            head_sample_img, c, _ = self.image_dataset.load_sample(head_sample)
            head_images.append(head_sample_img)
            head_labels.append(c)
        
        return tail_images, tail_labels, head_images, head_labels

    def collate_fn(self, batch):
        batch_tail_images = []
        batch_tail_labels = []
        batch_head_images = []
        batch_head_labels = []
        
        for tail_images, tail_labels, head_images, head_labels in batch:
            batch_tail_images += tail_images
            batch_tail_labels += tail_labels
            batch_head_images += head_images
            batch_head_labels += head_labels

        img = torch.stack(batch_tail_images+batch_head_images).contiguous()
        return img, batch_tail_labels, batch_head_labels
