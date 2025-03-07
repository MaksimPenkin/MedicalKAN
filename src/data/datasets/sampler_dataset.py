# """
# @author   Maksim Penkin
# """

from .. import samplers

from .base_dataset import IDataset


class SamplerDataset(IDataset):
    @property
    def sampler(self):
        return self._sampler

    def __init__(self, sampler, *args, **kwargs):
        super(SamplerDataset, self).__init__(*args, **kwargs)

        self._sampler = samplers.get(sampler)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
