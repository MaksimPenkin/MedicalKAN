# """
# @author   Maksim Penkin
# """

from src.utils.torch_utils import torch_load

from src.utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(identifier, **kwargs)

    if checkpoint:
        obj = torch_load(obj, checkpoint, strict=True)
    return obj
