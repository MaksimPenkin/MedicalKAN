# """
# @author   Maksim Penkin
# """

import inspect
from torch.nn import CrossEntropyLoss, MSELoss

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ce": CrossEntropyLoss,
                            "mse": MSELoss},
                        **kwargs)

    if callable(obj):
        if inspect.isclass(obj):
            return obj()
        else:
            return obj
    raise ValueError(f"Could not interpret loss instance: {obj}.")
