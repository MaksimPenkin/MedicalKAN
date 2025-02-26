# """
# @author   Maksim Penkin
# """

from .base_metric import IMetric
from .mean_metric import MeanMetric

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "MeanMetric": MeanMetric,
                            "CompositeMetric": CompositeMetric},
                        **kwargs)

    if isinstance(obj, IMetric):
        return obj
    raise ValueError(f"Could not interpret metric instance: {obj}.")


class CompositeMetric(IMetric):

    def __init__(self, metrics=None, name=None):
        super(CompositeMetric, self).__init__(name=name)

        if metrics:
            self.metrics = {metric.name: metric for metric in [get(m) for m in metrics]}
        else:
            self.metrics = {}

    def reset_state(self):
        for m in self.metrics.values():
            m.reset_state()

    def update_state(self, value, n=1):
        for k, v in value.items():
            if k not in self.metrics:
                self.metrics[k] = MeanMetric(name=k)
            self.metrics[k].update_state(v, n=n)

    def result(self):
        return {m.name: m.result() for m in self.metrics.values()}
