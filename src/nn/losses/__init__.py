# """
# @author   Maksim Penkin
# """

import inspect

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier, **kwargs)

    if callable(obj):
        if inspect.isclass(obj):
            return obj()
        else:
            return obj
    raise ValueError(f"Could not interpret loss instance: {obj}.")
