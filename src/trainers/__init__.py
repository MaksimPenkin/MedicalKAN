# """
# @author   Maksim Penkin
# """

from lightning import Trainer
from .base_trainer import CommonLitTrainer

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "LitTrainer": Trainer,
                            "CommonLitTrainer": CommonLitTrainer},
                        **kwargs)

    if isinstance(obj, Trainer):
        return obj
    raise ValueError(f"Could not interpret trainer instance: {obj}.")
