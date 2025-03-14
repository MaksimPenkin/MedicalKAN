# """
# @author   Maksim Penkin
# """

from .base_sampler import ISampler
from .source_sampler import TXTSampler, CSVSampler

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "TXTSampler": TXTSampler,
                            "CSVSampler": CSVSampler},
                        **kwargs)

    if isinstance(obj, ISampler):
        return obj
    raise ValueError(f"Could not interpret sampler instance: {obj}.")
