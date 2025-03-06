# """
# @author   Maksim Penkin
# """

import inspect
from torch.nn import CrossEntropyLoss, MSELoss

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "CrossEntropyLoss": CrossEntropyLoss,
                            "MSELoss": MSELoss,
                            "CompositeLoss": CompositeLoss},
                        **kwargs)

    if callable(obj):
        if inspect.isclass(obj):
            return obj()
        else:
            return obj
    raise ValueError(f"Could not interpret loss instance: {obj}.")


class CompositeLoss:

    def __init__(self, losses=None):
        if losses:
            self.losses = {loss.name: loss for loss in [get(l) for l in losses]}
        else:
            self.losses = {}

    def __call__(self, y_pred, y, **kwargs):
        total_loss = 0.0
        logs = {}
        for name, loss_fn in self.losses.items():
            v = loss_fn(y_pred, y, **kwargs)
            total_loss += v * loss_fn.weight
            logs[name] = v
        logs["loss"] = total_loss

        return logs
