# """
# @author   Maksim Penkin
# """

from data.samplers.base_sampler import PathSampler


class TXTSampler(PathSampler):

    def _set_data(self, filename):
        with open(filename, "rt") as f:
            lines = f.read().splitlines()
        lines = (line.rstrip() for line in lines)
        self._data = [line.split(" ") for line in lines if line]

    def _set_multiplicity(self):
        l1 = len(self.data[0])
        assert all(len(line) == l1 for line in self.data)
        self._multiplicity = l1

    def _get_item(self, item):
        return self.data[item]

    def __init__(self, *args, **kwargs):
        super(TXTSampler, self).__init__(*args, **kwargs)
