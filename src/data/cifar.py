# """
# @author   Maksim Penkin
# """

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.constants import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


def cifar10(root="/media/datasets/CIFAR10", split="val", **kwargs):
    db = datasets.CIFAR10(root, train=(split == "train"), download=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=CIFAR10_MEAN,
                                                                             std=CIFAR10_STD)]))
    return DataLoader(db, **kwargs)


def cifar100(root="/media/datasets/CIFAR100", split="val", **kwargs):
    db = datasets.CIFAR100(root, train=(split == "train"), download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(mean=CIFAR100_MEAN,
                                                                              std=CIFAR100_STD)]))
    return DataLoader(db, **kwargs)
