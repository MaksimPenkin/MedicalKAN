# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from callbacks.base_callback import ICallback
from callbacks.ckpt_callback import ModelCheckpointCallback
from callbacks.tb_callback import TensorBoardCallback

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "modelcheckpoint": ModelCheckpointCallback,
                            "tensorboard": TensorBoardCallback},
                        **kwargs)

    if isinstance(obj, ICallback):
        return obj
    raise ValueError(f"Could not interpret callback instance: {obj}.")
