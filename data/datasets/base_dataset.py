# """
# @author   Maksim Penkin
# """

from data.transforms import CompositeTransform

from torch.utils.data import Dataset


class CustomDataset(Dataset):

    @property
    def root(self):
        return self._root

    @property
    def transforms(self):
        return self._transforms

    def __init__(self, root, transforms=None):
        self._root = root
        self._transforms = CompositeTransform(transforms) if transforms is not None else None

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclasses.")
