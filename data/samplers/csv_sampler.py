# """
# @author   Maksim Penkin
# """

import os
import pandas as pd
from data.samplers.base_sampler import IterableSampler


class CSVSampler(IterableSampler):

    @property
    def file(self):
        return self._file

    def __init__(self, filename, root="", with_names=False):
        super(CSVSampler, self).__init__()

        self._file = pd.read_csv(filename)
        self._multiplicity = len(self.file.columns.values.tolist())
        self._root = root
        self._with_names = bool(with_names)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, item):
        filenames = self.file.iloc[item].values.tolist()

        sample = tuple(os.path.join(self._root, filename) for filename in filenames)
        if self._with_names:
            sample += (os.path.split(filenames[0])[-1], )

        return sample
