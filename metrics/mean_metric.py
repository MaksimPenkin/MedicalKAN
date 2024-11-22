# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from metrics.base_metric import IMetric


class MeanMetric(IMetric):

    @property
    def total(self):
        return self._total

    @property
    def count(self):
        return self._count

    def __init__(self, *args, **kwargs):
        super(MeanMetric, self).__init__(*args, **kwargs)

        self._total = 0.0
        self._count = 0

    def reset_state(self):
        self._total = 0.0
        self._count = 0

    def update_state(self, value, n=1):
        self._total += value * n
        self._count += n

    def result(self):
        if self._count == 0:
            return 0.0
        else:
            return self._total / self._count
