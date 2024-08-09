# """
# @author   Maksim Penkin
# """

from data.samplers.base_sampler import IterableSampler


class TextLineSampler(IterableSampler):

    @property
    def file(self):
        return self._file

    def __init__(self, filename):
        super(TextLineSampler, self).__init__()

        with open(filename, "rt") as f:
            self._file = f.read().splitlines()

    def __len__(self):
        return len(self.file)

    def __getitem__(self, item):
        super(TextLineSampler, self).__getitem__(item)

        return self.file[item]
