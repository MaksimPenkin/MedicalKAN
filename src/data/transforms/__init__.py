# """
# @author   Maksim Penkin
# """

from .totensor_transform import ToTensor

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ToTensor": ToTensor,
                            "CompositeTransform": CompositeTransform},
                        **kwargs)

    if callable(obj):
        return obj
    raise ValueError(f"Could not interpret transform instance: {obj}.")


class CompositeTransform:

    def __init__(self, transforms=None):
        if transforms:
            self.transforms = [get(t) for t in transforms]
        else:
            self.transforms = []

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
