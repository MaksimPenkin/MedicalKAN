# """
# @author   Maksim Penkin
# """

from utils.torch_utils import torch_random

from data.samplers.base_sampler import IterableSampler


class UniformSampler(IterableSampler):

    @property
    def parameters(self):
        return self._parameters

    def __init__(self, parameters=None, n=1):
        super(UniformSampler, self).__init__()

        self._parameters = parameters if parameters is not None else []
        assert n >= 0, f"The number of items in a collection can't be negative, found: {n}."
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return tuple(torch_random(**kwargs) for kwargs in self.parameters)
