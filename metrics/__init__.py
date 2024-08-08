# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from tools.optimization.metrics.base_metric import IMetric, MeanMetric

from tools.optimization.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "MeanMetric": MeanMetric},
                        **kwargs)

    if isinstance(obj, IMetric):
        return obj
    raise ValueError(f"Could not interpret metric instance: {obj}.")
