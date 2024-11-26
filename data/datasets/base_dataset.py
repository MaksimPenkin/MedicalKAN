# """
# @author   Maksim Penkin
# """

from data import samplers, transforms

from torch.utils.data import Dataset


class SampleDataset(Dataset):

    @property
    def sampler(self):
        return self._sampler

    @property
    def transform(self):
        return self._transform

    def __init__(self, sampler, transform=None):
        self._sampler = samplers.get(sampler)
        if transform is not None:
            self._transform = transforms.CompositeTransform([transforms.get(t) for t in transform])
        else:
            self._transform = None

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler[idx]
        if self.transform:
            sample = self.transform(*sample)
        return sample
