# """
# @author   Maksim Penkin
# """

from . import datasets
import albumentations as A
from torch.utils.data import DataLoader


def ixi(dataset, **kwargs):
    transform = A.Compose([
        A.ToTensorV2(transpose_mask=True)
    ])

    db = datasets.get(dataset, transform=transform)
    return DataLoader(db, **kwargs)
