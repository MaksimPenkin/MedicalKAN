# """
# @author   Maksim Penkin
# """

from .ckan.ckan import ConvKAN
from .resnet.resnet import resnet18, resnet50, resnet101
from .srkan.srkan import StackedResidualKAN
from .unet.unet import UNet
from src.utils.torch_utils import torch_load

from src.utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ckan": ConvKAN,
                            "resnet18": resnet18,
                            "resnet50": resnet50,
                            "resnet101": resnet101,
                            "srkan": StackedResidualKAN,
                            "unet": UNet},
                        **kwargs)

    if checkpoint:
        obj = torch_load(obj, checkpoint, strict=True)
    return obj
