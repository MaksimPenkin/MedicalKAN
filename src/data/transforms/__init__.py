# """
# @author   Maksim Penkin
# """

from .totensor_transform import ToTensor

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier, **kwargs)

    if callable(obj):
        return obj
    raise ValueError(f"Could not interpret transform instance: {obj}.")


class Compose:

    def __init__(self, transforms=None):
        if transforms:
            self.transforms = [get(t) for t in transforms]
        else:
            self.transforms = []

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
