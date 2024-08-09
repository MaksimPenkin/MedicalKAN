# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

from torch.utils.data import DataLoader
from data.datasets.dummy_dataset import RandomUniformDataset
from data.datasets.image_dataset import ImageDataset

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "random_uniform": RandomUniformDataset,
                            "image": ImageDataset
                        },
                        **kwargs)

    if isinstance(obj, DataLoader):
        return obj
    raise ValueError(f"Could not interpret dataset instance: {obj}.")
