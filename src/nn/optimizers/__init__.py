# """
# @author   Maksim Penkin
# """

from torch.optim import Optimizer, Adadelta, Adagrad, Adam, RMSprop, SGD

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "adadelta": Adadelta,
                            "adagrad": Adagrad,
                            "adam": Adam,
                            "rmsprop": RMSprop,
                            "sgd": SGD},
                        **kwargs)

    if isinstance(obj, Optimizer):
        return obj
    raise ValueError(f"Could not interpret optimizer instance: {obj}.")
