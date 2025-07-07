# """
# @author   Maksim Penkin
# """

from . import datasets
import albumentations as A
from torch.utils.data import DataLoader


def busi(dataset, split="val", **kwargs):
    if split == "train":
        transform = A.Compose([
            A.Resize(256, 256),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(p=0.5),
            A.ToTensorV2(transpose_mask=True)
        ])
    elif split == "val":
        transform = A.Compose([
            A.Resize(256, 256),
            A.ToTensorV2(transpose_mask=True)
        ])
    else:
        raise ValueError(f"Unrecognized `split` found: {split}.")

    db = datasets.get(dataset, transform=transform)
    return DataLoader(db, **kwargs)
