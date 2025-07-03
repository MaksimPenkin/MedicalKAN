# """
# @author   Maksim Penkin
# """

import inspect
import torch.nn as nn
from .dice_loss import DiceWithLogitsLoss

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "DiceWithLogitsLoss": DiceWithLogitsLoss},
                        **kwargs)

    if callable(obj):
        if inspect.isclass(obj):
            return obj()
        else:
            return obj
    raise ValueError(f"Could not interpret loss instance: {obj}.")


class WeightedLoss(nn.Module):

    @property
    def weight(self):
        return self._weight

    def __init__(self, loss):
        super(WeightedLoss, self).__init__()

        self.loss = get(loss)

    def forward(self, *args, **kwargs):
        return self.weight * self.loss(*args, **kwargs)


class CompositeLoss(nn.Module):

    def __init__(self, losses=None):
        super(CompositeLoss, self).__init__()

        if losses:
            self.losses = {l_name: get(l_cfg) for l_name, l_cfg in losses}
        else:
            self.losses = {}

    def forward(self, *args, **kwargs):
        total_loss = 0.0
        logs = {}
        for loss_name, loss_fn in self.losses.items():
            v = loss_fn(*args, **kwargs)
            total_loss += v
            logs[loss_name] = v
        logs["loss"] = total_loss

        return logs
