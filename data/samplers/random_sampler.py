# """
# @author   Maksim Penkin
# """

from itertools import zip_longest
from utils.torch_utils import torch_random

from data.samplers.base_sampler import IterableSampler


class UniformSampler(IterableSampler):

    @property
    def shapes(self):
        return self._shapes

    @property
    def parameters(self):
        return self._parameters

    @property
    def dtypes(self):
        return self._dtypes

    def __init__(self, shapes, parameters=None, dtypes=None, n=1):
        super(UniformSampler, self).__init__()

        self._shapes = shapes
        self._parameters = parameters
        self._dtypes = dtypes

        assert n >= 0, f"The number of items in a collection can't be negative, found: {n}."
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return tuple(
            torch_random(shape, low=low, high=high, dtype=dtype)
            for shape, (low, high), dtype in zip_longest(self.shapes, self.parameters, self.dtypes)
        )
