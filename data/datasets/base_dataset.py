# """
# @author   Maksim Penkin
# """

from data.transforms import CompositeTransform

from torch.utils.data import Dataset


class IDataset(Dataset):

    @property
    def transforms(self):
        return self._transforms

    def __init__(self, transforms=None):
        self._transforms = CompositeTransform(transforms) if transforms is not None else None

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclasses.")
