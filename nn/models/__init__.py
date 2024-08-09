# """
# @author   Maksim Penkin
# """

import torch
from nn.models.resnet import resnet18, resnet50, resnet101

from utils.serialization_utils import create_object


def get(identifier, checkpoint=None, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "resnet18": resnet18,
                            "resnet50": resnet50,
                            "resnet101": resnet101},
                        **kwargs)

    if isinstance(obj, torch.nn.Module):
        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                state_dict = {name[6:]: checkpoint[name] for name in checkpoint}
            else:
                state_dict = checkpoint
            obj.load_state_dict(state_dict, strict=False)
        return obj
    raise ValueError(f"Could not interpret model instance: {obj}.")
