# """
# @author   Maksim Penkin
# """

from data import samplers

from data.datasets.base_dataset import CustomDataset


class SamplerDataset(CustomDataset):

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
        if self.transforms:
            sample = self.transforms(sample)
        return sample
