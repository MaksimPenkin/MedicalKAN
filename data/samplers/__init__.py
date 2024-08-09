# """
# @author   Maksim Penkin
# """

from data.samplers.base_sampler import ISampler
from data.samplers.csv_sampler import CSVSampler
from data.samplers.txt_sampler import TXTSampler

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "csv": CSVSampler,
                            "txt": TXTSampler
                        },
                        **kwargs)

    if isinstance(obj, ISampler):
        return obj
    raise ValueError(f"Could not interpret sampler instance: {obj}.")
