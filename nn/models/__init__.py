# """
# @author   Maksim Penkin
# """

import torch
from nn.models.ckan import ConvKAN
from nn.models.ukan import StackedResidualKAN
from nn.models.resnet import resnet18, resnet50, resnet101
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

    if isinstance(obj, torch.nn.Module):
        if checkpoint:
            obj = torch_load(obj, checkpoint, strict=False)
        return obj
    raise ValueError(f"Could not interpret model instance: {obj}.")
