# """
# @author   Maksim Penkin
# """

from data import samplers
from utils.io_utils import read_img

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, sampler, transform=None, **kwargs):
        self.sampler = samplers.get(sampler)
        self.transform = transform
        self._kwargs = kwargs

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        args = self.sampler[idx]
        t = min(len(args), self.sampler.multiplicity)

        sample = tuple(read_img(args[i], **{k: v[i] for k, v in self._kwargs.items()}) for i in range(t))
        if self.transform:
            sample = self.transform(sample)

        return *sample, *args[t:]
