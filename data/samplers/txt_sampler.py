# """
# @author   Maksim Penkin
# """

from data.samplers.base_sampler import PathSampler


class TXTSampler(PathSampler):

    def __init__(self, *args, **kwargs):
        super(TXTSampler, self).__init__(*args, **kwargs)

    def _set_data(self, filename):
        with open(filename, "rt") as f:
            lines = f.read().splitlines()
        lines = (line.rstrip() for line in lines)
        self._data = [line.split(" ") for line in lines if line]

    def _get_data_item(self, item):
        return self.data[item]
