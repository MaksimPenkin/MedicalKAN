# """
# @author   Maksim Penkin
# """

from data.transforms.base_transform import ITransform
from data.transforms.totensor_transform import ToTensor

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ToTensor": ToTensor},
                        **kwargs)

    if isinstance(obj, ITransform):
        return obj
    raise ValueError(f"Could not interpret transform instance: {obj}.")


class CompositeTransform(ITransform):

    def __init__(self, transforms=None):
        super(CompositeTransform, self).__init__()

        if transforms:
            self.transforms = [get(t) for t in transforms]
        else:
            self.transforms = []

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
