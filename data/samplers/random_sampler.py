# """
# @author   Maksim Penkin
# """

from utils.torch_utils import torch_random

from data.samplers.base_sampler import IterableSampler


class UniformSampler(IterableSampler):

    @property
    def random_vector(self):
        return self._random_vector

    def __init__(self, random_vector, n=1):
        super(UniformSampler, self).__init__()

        assert isinstance(random_vector, (list, tuple))
        self._random_vector = random_vector

        assert isinstance(n, int) and n >= 0, f"The number of items in a collection can't be negative or non-integer, found: {n}."
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, item):
        return tuple(torch_random(**kwargs) for kwargs in self.random_vector)
