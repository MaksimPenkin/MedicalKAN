# """
# @author   Maksim Penkin
# """

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.constants import MNIST_MEAN, MNIST_STD


def mnist(root="/media/datasets/MNIST", split="val", **kwargs):
    db = torchvision.datasets.MNIST(root, train=(split == "train"), download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(mean=MNIST_MEAN,
                                                                                       std=MNIST_STD)]))
    return DataLoader(db, **kwargs)
