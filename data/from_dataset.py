# """
# @author   Maksim Penkin
# """

from data import datasets
from torch.utils.data import DataLoader


def from_dataset(dataset, **kwargs):
    db = datasets.get(dataset)
    return DataLoader(db, **kwargs)
