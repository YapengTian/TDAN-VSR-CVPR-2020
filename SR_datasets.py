from __future__ import division
import torchvision
import torchvision.transforms as T
import os
import glob
import scipy.misc
import scipy.ndimage
from PIL import Image, ImageFile
import numpy as np
import h5py
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VSR_Dataset(object):
    def __init__(self, dir, trans = None):

        self.dir_HR = os.path.join(dir, "HR")
        self.dir_LR = os.path.join(dir, "LR_bicubic")
        self.lis = sorted(os.listdir(self.dir_HR))

        self.transform = trans

    def __len__(self):
        return len(self.lis)

    def __getitem__(self, idx):
        HR = os.path.join(self.dir_HR, self.lis[idx])
        LR = os.path.join(self.dir_LR, self.lis[idx])
        scale = 4
        ims = sorted(os.listdir(HR))
        # get frame size
        image = io.imread(os.path.join(HR, ims[0]))
        row, col, ch = image.shape
        frames_lr = np.zeros((5, int(row / scale), int(col / scale), ch))
        if len(ims) > 5:
            center = random.randint(2, len(ims)-3)
        else:
            center = len(ims) // 2
        frames_hr = io.imread(os.path.join(HR, ims[center]))

        for j in range(center - 2, center + 3):  # only use 5 frames
            i = j - center + 2
            frames_lr[i, :, :, :] = io.imread(os.path.join(LR, ims[j]))

        sample = {'lr': frames_lr, 'hr': frames_hr, 'im_name': ims[center]}

        if self.transform:
            sample = self.transform(sample)
        return sample

class DataAug(object):
    def __call__(self, sample):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        lr, hr, name = sample['lr'], sample['hr'], sample['im_name']
        num, r, c, ch = lr.shape

        if hflip:
            hr = hr[:, ::-1, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, :, ::-1, :]
        if vflip:
            hr = hr[::-1, :, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, ::-1, :, :]
        if rot90:
            hr= hr.transpose(1, 0, 2)
            lr = lr.transpose(0, 2, 1, 3)

        return {'lr': lr, 'hr': hr, 'im_name':name}


class RandomCrop(object):

    """crop randomly the image in a sample
    Args, output_size:desired output size. If int, square crop is mad

    """
    def __init__(self, output_size, scale):
        self.scale = scale
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        lr, hr, name = sample['lr'], sample['hr'], sample['im_name']
        h, w = lr.shape[1: 3]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_lr = lr[:,top:top + new_h, left: left + new_w, :]
        new_hr = hr[top*self.scale:top*self.scale + new_h*self.scale, left*self.scale: left*self.scale + new_w*self.scale, :]

        return {'lr': new_lr, "hr": new_hr, "im_name": name}

class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name = sample['lr']/255.0 - 0.5, sample['hr']/255.0 - 0.5, sample['im_name']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'im_name':name}
