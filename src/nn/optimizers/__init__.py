# """
# @author   Maksim Penkin
# """

import functools
from torch.optim import Optimizer

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier, **kwargs)

    if isinstance(obj, Optimizer) or (isinstance(obj, functools.partial) and issubclass(obj.func, Optimizer)):
        return obj
    raise ValueError(f"Could not interpret optimizer instance: {obj}.")
