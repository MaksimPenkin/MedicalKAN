# """
# @author   Maksim Penkin
# """

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.constants import IMAGENET_MEAN, IMAGENET_STD


def imagenet1k(root="/media/datasets/ILSVRC2012", split="val", **kwargs):
    db = datasets.ImageFolder(os.path.join(root, split),
                              transform=transforms.Compose([transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=IMAGENET_MEAN,
                                                                                 std=IMAGENET_STD)]))
    return DataLoader(db, **kwargs)
