# """
# @author   Maksim Penkin
# """

from .base_engine import IEngine
from .train_engine import Trainer

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "Trainer": Trainer
                        },
                        **kwargs)

    if isinstance(obj, IEngine):
        return obj
    raise ValueError(f"Could not interpret engine instance: {obj}.")
