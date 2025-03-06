# """
# @author   Maksim Penkin
# """

from .. import transforms

from torch.utils.data import Dataset


class IDataset(Dataset):

    @property
    def transform(self):
        return self._transform

    def __init__(self, transform=None):
        # The transforms must be designed to fit the dataset.
        # As such, the dataset must output a sample compatible with the library transform functions,
        # or transforms must be defined for the particular sample case.
        self._transform = transforms.get(transform) if transform is not None else None

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclasses.")
