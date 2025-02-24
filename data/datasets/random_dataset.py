# """
# @author   Maksim Penkin
# """

from itertools import zip_longest
from utils.torch_utils import torch_dtype, torch_random

from torch.utils.data import Dataset


class RandomUniformDataset(Dataset):

    @property
    def shapes(self):
        return self._shapes

    @property
    def dtypes(self):
        return self._dtypes

    def __init__(self, *shapes, dtypes=None, n=1):
        self._shapes = shapes

        if isinstance(dtypes, (list, tuple)):
            self._dtypes = tuple(torch_dtype(x) for x in dtypes)
        else:
            self._dtypes = (torch_dtype(dtypes), )

        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return tuple(torch_random(shape, dtype=dtype) for shape, dtype in zip_longest(self.shapes, self.dtypes))
