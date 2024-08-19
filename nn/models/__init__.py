# """
# @author   Maksim Penkin
# """

import torch
from nn.models.ckan import ConvKANv0
from nn.models.resnet import resnet18, resnet50, resnet101

from utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "convkanv0": ConvKANv0,
                            "resnet18": resnet18,
                            "resnet50": resnet50,
                            "resnet101": resnet101},
                        **kwargs)

    if isinstance(obj, torch.nn.Module):
        if checkpoint:
            obj.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return obj
    raise ValueError(f"Could not interpret model instance: {obj}.")
