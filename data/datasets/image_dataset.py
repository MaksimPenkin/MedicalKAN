# """
# @author   Maksim Penkin
# """

from data import samplers
from data.samplers.base_sampler import PathSampler
from utils.io_utils import read_img

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, sampler, transform=None, **kwargs):
        self.sampler = samplers.get(sampler)
        assert isinstance(self.sampler, PathSampler), f"Error: expected `sampler` PathSampler, found: {self.sampler}."
        self.transform = transform
        self._kwargs = kwargs

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        paths = self.sampler[idx]
        t = len(paths) - self.sampler.with_names

        sample = tuple(read_img(paths[i], **{k: v[i] for k, v in self._kwargs.items()}) for i in range(t))
        if self.transform:
            sample = self.transform(sample)

        return *sample, *paths[t:]
