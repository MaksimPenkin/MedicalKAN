# """
# @author   Maksim Penkin
# """

import os
from data.samplers.base_sampler import IterableSampler


class TXTSampler(IterableSampler):

    @property
    def file(self):
        return self._file

    def __init__(self, filename, root="", with_names=False):
        super(TXTSampler, self).__init__()

        with open(filename, "rt") as f:
            self._file = f.read().splitlines()
        self._root = root
        self._with_names = bool(with_names)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, item):
        filenames = self.file[item].split(" ")

        sample = tuple(os.path.join(self._root, filename) for filename in filenames)
        if self._with_names:
            sample += (os.path.split(filenames[0])[-1], )

        return sample
