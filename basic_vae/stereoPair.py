import os
import numpy as np
from torch.utils import data
from PIL import Image


class stereoPair(data.Dataset):
    def __init__(self, path, device, is_semi=False, transform=None):
        self.transform = transform
        left_path = os.path.join(path, 'input')
        label_path = os.path.join(path, 'label')
        self.left_imgs = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if f.endswith('.jpg')])
        self.left_imgs = [Image.open(f) for f in self.left_imgs]
        if is_semi:
            self.labels = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.cat')])
            assert len(self.left_imgs) == len(self.labels)
            self.labels = [Image.open(f) for f in self.labels]
        self.transform = transform
        self.is_semi = is_semi
        self.device = device
        

    def __getitem__(self, index):
        img = self.left_imgs[index]
        if self.is_semi:
            gt = self.right_imgs[index]
            if self.transform:
                img = self.transform(img)
                gt = self.transform(gt)
            return img.to(self.device), gt.to(self.device)
        else:
        if self.transform:
            img = self.transform(img)
        return img.to(self.device)

    def __len__(self):
        return len(self.left_imgs)

