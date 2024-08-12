# """
# @author   Maksim Penkin
# """

import numpy as np
import torch

from data.datasets.image_dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader


class ToTensor:

    def __call__(self, sample):
        return tuple(torch.from_numpy(np.transpose(x[..., np.newaxis], axes=(2, 0, 1))) for x in sample)


def ixi(sampler, key=("sketch", "gt"), **kwargs):
    db = ImageDataset(sampler,
                      transform=transforms.Compose([ToTensor()]),
                      key=key)
    return DataLoader(db, **kwargs)
