# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

import inspect
from torch.nn import CrossEntropyLoss

from tools.optimization.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "CrossEntropyLoss": CrossEntropyLoss},
                        **kwargs)

    if callable(obj):
        if inspect.isclass(obj):
            return obj()
        else:
            return obj
    raise ValueError(f"Could not interpret loss instance: {obj}.")
