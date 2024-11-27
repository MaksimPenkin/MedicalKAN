# """
# @author   Maksim Penkin
# """

import os
from itertools import zip_longest
from utils.io_utils import read_img

from data.datasets.base_dataset import SamplerDataset


class ImageDataset(SamplerDataset):

    def __init__(self, *args, load_func=None, load_params=None, return_filenames=False, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

        self._load_func = load_func or read_img
        self._load_params = load_params or []
        self.return_filenames = bool(return_filenames)

    def __getitem__(self, index):
        filenames = self.sampler[index]

        sample = tuple(
            self._load_func(os.path.join(self.root, filename), **kwargs)
            for filename, kwargs in zip_longest(filenames, self._load_params, fillvalue={})
        )

        if self.transforms:
            sample = self.transforms(*sample)

        if self.return_filenames:
            return *sample, os.path.split(filenames[0])[-1]
        else:
            return sample
