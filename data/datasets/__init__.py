# """
# @author   Maksim Penkin
# """

from torch.utils.data import Dataset
from data.datasets.base_dataset import SampleDataset
from data.datasets.image_dataset import ImageDataset

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "SampleDataset": SampleDataset,
                            "ImageDataset": ImageDataset},
                        **kwargs)

    if isinstance(obj, Dataset):
        return obj
    raise ValueError(f"Could not interpret dataset instance: {obj}.")
