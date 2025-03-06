# """
# @author   Maksim Penkin
# """

from lightning.pytorch.loggers import Logger

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier, **kwargs)

    if isinstance(obj, Logger):
        return obj
    raise ValueError(f"Could not interpret logger instance: {obj}.")
