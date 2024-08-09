# """
# @author   Maksim Penkin
# """

from data.datasets.image_dataset import ImageDataset
from data.samplers.csv_sampler import CSVSampler
from torchvision import transforms

import numpy as np
import torch
from torch.utils.data import DataLoader


class ToTensor:

    def __call__(self, sample):
        return tuple(torch.from_numpy(np.transpose(x[..., np.newaxis], axes=(2, 0, 1))) for x in sample)


def ixi(filename, root="", key=("sketch", "gt"), **kwargs):
    db = ImageDataset(CSVSampler(filename, root=root),
                      transform=transforms.Compose([ToTensor()]),
                      key=key)

    return DataLoader(db, **kwargs)
