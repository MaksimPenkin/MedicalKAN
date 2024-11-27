# """
# @author   Maksim Penkin
# """

import os
from utils.io_utils import read_img
from utils.serialization_utils import create_func

from data.datasets.base_dataset import SamplerDataset


class ImageDataset(SamplerDataset):

    @property
    def loader(self):
        return self._load_func

    def __init__(self, *args, load_func=None, load_params=None, return_filenames=False, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

        self._load_func = create_func(load_func) or read_img
        self._load_params = load_params
        self.return_filenames = bool(return_filenames)

    def __getitem__(self, index):
        filenames = self.sampler[index]
        params = self._load_params or [{}] * len(filenames)

        sample = tuple(
            self.loader(os.path.join(self.root, filename), **kwargs)
            for filename, kwargs in zip(filenames, params)
        )

        if self.transforms:
            sample = self.transforms(*sample)

        if self.return_filenames:
            return *sample, os.path.split(filenames[0])[-1]
        else:
            return sample
