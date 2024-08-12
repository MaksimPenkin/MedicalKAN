# """
# @author   Maksim Penkin
# """

import pandas as pd

from data.samplers.base_sampler import PathSampler


class CSVSampler(PathSampler):

    def __init__(self, *args, **kwargs):
        super(CSVSampler, self).__init__(*args, **kwargs)

    def _set_data(self, filename):
        self._data = pd.read_csv(filename)

    def _get_data_item(self, item):
        return self.data.iloc[item].values.tolist()
