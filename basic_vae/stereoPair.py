import os
import numpy as np
from torch.utils import data
from PIL import Image


class stereoPair(data.Dataset):
    def __init__(self, path, device, is_training=True, transform=None):
        self.transform = transform
        left_path = os.path.join(path, 'input')
        right_path = os.path.join(path, 'gt')
        self.left_imgs = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if f.endswith('png')])
        self.left_imgs = [Image.open(f) for f in self.left_imgs]
        # if is_training:
        #     self.right_imgs = sorted([os.path.join(right_path, f) for f in os.listdir(right_path) if f.endswith('png')])
        #     assert len(self.left_imgs) == len(self.right_imgs)
        #     self.right_imgs = [Image.open(f) for f in self.right_imgs]
        self.transform = transform
        self.is_training = is_training
        self.device = device
        

    def __getitem__(self, index):
        img = self.left_imgs[index]
        # if self.is_training:
        #     gt = self.right_imgs[index]
        #     if self.transform:
        #         img = self.transform(img)
        #         gt = self.transform(gt)
        #     return img.to(self.device), gt.to(self.device)
        # else:
        if self.transform:
            img = self.transform(img)
        return img.to(self.device)

    def __len__(self):
        return len(self.left_imgs)

