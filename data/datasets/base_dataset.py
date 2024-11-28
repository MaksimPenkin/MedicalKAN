# """
# @author   Maksim Penkin
# """

from data import transforms as augmentations

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

        if transforms is not None:
            self._transforms = augmentations.CompositeTransform([augmentations.get(t) for t in transforms])
        else:
            self._transforms = None

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclasses.")
