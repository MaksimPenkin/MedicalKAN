# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from torch.optim.lr_scheduler import LRScheduler, LinearLR, ExponentialLR, PolynomialLR, CyclicLR

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "linear": LinearLR,
                            "exponential": ExponentialLR,
                            "polynomial": PolynomialLR,
                            "cyclic": CyclicLR},
                        **kwargs)

    if isinstance(obj, LRScheduler):
        return obj
    raise ValueError(f"Could not interpret scheduler instance: {obj}.")
