# """
# @author   Maksim Penkin
# """

from data.samplers.path_sampler import PathSampler
from utils.io_utils import read_img

from data.datasets.base_dataset import SampleDataset


class ImageDataset(SampleDataset):

    def __init__(self, sampler, transform=None, **kwargs):
        super(ImageDataset, self).__init__(sampler, transform=transform)

        assert isinstance(self.sampler, PathSampler), f"Error: expected `sampler` PathSampler, found: {self.sampler}."
        self._kwargs = kwargs

    def __getitem__(self, idx):
        paths = self.sampler[idx]
        t = len(paths) - self.sampler.with_names

        sample = tuple(read_img(paths[i], **{k: v[i] for k, v in self._kwargs.items()}) for i in range(t))
        if self.transform:
            sample = self.transform(*sample)

        return *sample, *paths[t:]
