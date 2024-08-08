"""
 @author   Maksim Penkin

"""

import os

import pandas as pd
import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ToTensor(object):

    def __call__(self, sample):
        sketch, gt = sample[0], sample[1]

        # Swap color axis H x W x C -> C X H X W
        sketch = sketch.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))

        return torch.from_numpy(sketch), torch.from_numpy(gt)


class MatDataset(Dataset):
    """
        Args:
            csv_file (string): Path to the *.csv file with images and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        if not transform:
            self.transform = transforms.Compose([ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        sketch_name = os.path.join(self.root_dir,
                                   self.csv_data.iloc[idx, 0])
        gt_name = os.path.join(self.root_dir,
                               self.csv_data.iloc[idx, 1])

        sketch = loadmat(sketch_name, appendmat=False)["sketch"]
        sketch = sketch[..., np.newaxis].astype(np.float32)

        gt = loadmat(gt_name, appendmat=False)["image"]
        gt = gt[..., np.newaxis].astype(np.float32)

        sample = (sketch, gt)

        if self.transform:
            sample = self.transform(sample)

        return sample
