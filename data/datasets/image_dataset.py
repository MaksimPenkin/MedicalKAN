# """
# @author   Maksim Penkin
# """

import os
from itertools import zip_longest
from utils.io_utils import read_img

from data.datasets.base_dataset import SamplerDataset


class ImageDataset(SamplerDataset):

    def __init__(self, *args, load_params=None, return_filename=False, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

        self.load_func = read_img  # Hard-coded loader.
        self.load_params = load_params if load_params is not None else []
        self.return_filename = bool(return_filename)

    def __getitem__(self, idx):
        filenames = self.sampler[idx]

        sample = tuple(
            self.load_func(os.path.join(self.root, filename), **kwargs)
            for filename, kwargs in zip_longest(filenames, self.load_params, fillvalue={})
        )

        if self.transforms:
            sample = self.transforms(*sample)

        if self.return_filename:
            return *sample, os.path.split(filenames[0])[-1]
        else:
            return sample
