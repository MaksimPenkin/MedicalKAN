# """
# @author   Maksim Penkin

# """

import os
from data.samplers.textline_sampler import TextLineSampler


class TextPathSampler(TextLineSampler):

    def _assert_file(self):
        f = (x.split(" ") for x in self.file)
        l1 = len(next(f))
        if not all(len(s) == l1 for s in f):
            raise ValueError("data/samplers/textpath_sampler.py: class TextPathSampler: def _assert_file(...): "
                             f"error: expected all lines of file contain the same number `{l1}` of filenames, split by ' '.")
        return l1

    @property
    def multiplicity(self):
        return self._multiplicity

    def __init__(self, *args, basedir="", with_names=False, **kwargs):
        super(TextPathSampler, self).__init__(*args, **kwargs)

        self._multiplicity = self._assert_file()
        self._basedir = basedir
        self._with_names = bool(with_names)

    def __getitem__(self, item):
        line = super(TextPathSampler, self).__getitem__(item)
        filenames = line.split(" ")

        sample = tuple(os.path.join(self._basedir, filename) for filename in filenames)
        if self._with_names:
            sample += (os.path.split(filenames[0])[-1], )

        return sample
