# """
# @author   Maksim Penkin
# """

from nn.base_engine import IEngine

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={},
                        **kwargs)

    if isinstance(obj, IEngine):
        return obj
    raise ValueError(f"Could not interpret engine instance: {obj}.")
