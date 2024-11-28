# """
# @author   Maksim Penkin
# """

from torch.utils.data import Dataset
from data.datasets.sampler_dataset import SamplerDataset
from data.datasets.file_dataset import FileDataset
from data.datasets.ixi_dataset import IXIRingingDataset

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "SamplerDataset": SamplerDataset,
                            "FileDataset": FileDataset,
                            "IXIRingingDataset": IXIRingingDataset},
                        **kwargs)

    if isinstance(obj, Dataset):
        return obj
    raise ValueError(f"Could not interpret dataset instance: {obj}.")
