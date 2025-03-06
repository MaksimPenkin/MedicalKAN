# """
# @author   Maksim Penkin
# """

from lightning.pytorch.callbacks import Callback

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier, **kwargs)

    if isinstance(obj, Callback):
        return obj
    raise ValueError(f"Could not interpret callback instance: {obj}.")
