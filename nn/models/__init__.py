# """
# @author   Maksim Penkin
# """

from nn.models.ckan import ConvKAN
from nn.models.ukan import StackedResidualKAN
from nn.models.resnet.resnet import resnet18, resnet50, resnet101
from utils.torch_utils import torch_load

from utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ckan": ConvKAN,
                            "srkan": StackedResidualKAN,
                            "resnet18": resnet18,
                            "resnet50": resnet50,
                            "resnet101": resnet101},
                        **kwargs)

    if checkpoint:
        obj = torch_load(obj, checkpoint, strict=True)
    return obj
