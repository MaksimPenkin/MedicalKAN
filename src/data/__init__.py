# """
# @author   Maksim Penkin
# """

from torch.utils.data import DataLoader
from .mnist import mnist
from .cifar import cifar10, cifar100
from .imagenet import imagenet1k
from .dummy import random_uniform
from .from_dataset import from_dataset

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "mnist": mnist,
                            "cifar10": cifar10,
                            "cifar100": cifar100,
                            "imagenet1k": imagenet1k,
                            "random_uniform": random_uniform,
                            "from_dataset": from_dataset},
                        **kwargs)

    if isinstance(obj, DataLoader):
        return obj
    raise ValueError(f"Could not interpret data instance: {obj}.")
