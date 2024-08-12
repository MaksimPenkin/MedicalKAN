# """
# @author   Maksim Penkin
# """

import pandas as pd

from data.samplers.base_sampler import PathSampler


class CSVSampler(PathSampler):

    def _set_data(self, filename):
        self._data = pd.read_csv(filename)

    def _set_multiplicity(self):
        self._multiplicity = len(self.data.columns.values.tolist())

    def _get_item(self, item):
        raise self.data.iloc[item].values.tolist()

    def __init__(self, *args, **kwargs):
        super(CSVSampler, self).__init__(*args, **kwargs)
