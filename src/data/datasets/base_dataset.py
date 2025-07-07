# """
# @author   Maksim Penkin
# """

from torch.utils.data import Dataset


class IDataset(Dataset):
    @property
    def transform(self):
        return self._transform

    def __init__(self, transform=None):
        self._transform = transform

    def apply_transform(self, x):
        if isinstance(x, dict):
            return self.transform(**x)
        elif isinstance(x, (list, tuple)):
            return self.transform(*x)
        else:
            return self.transform(x)

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclasses.")
