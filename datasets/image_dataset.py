# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from torch.utils.data import Dataset
from utils.io_utils import read_img


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

        return *(read_img(args[i], **{k: v[i] for k, v in self._kwargs.items()}) for i in range(t)), *args[t:]
