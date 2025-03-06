# """
# @author   Maksim Penkin
# """

from torch.utils.data import Dataset
from .file_dataset import FileDataset
from .random_dataset import RandomUniformDataset
from .sampler_dataset import SamplerDataset

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "FileDataset": FileDataset,
                            "RandomUniformDataset": RandomUniformDataset,
                            "SamplerDataset": SamplerDataset},
                        **kwargs)

    if isinstance(obj, Dataset):
        return obj
    raise ValueError(f"Could not interpret dataset instance: {obj}.")
