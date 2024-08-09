# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from torch.utils.data import DataLoader
from datasets.imagenet import imagenet1k
from datasets.dummy import random_uniform

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "imagenet1k": imagenet1k,
                            "random_uniform": random_uniform},
                        **kwargs)

    if isinstance(obj, DataLoader):
        return obj
    raise ValueError(f"Could not interpret dataset instance: {obj}.")
