# """
# @author   Maksim Penkin
# """

import abc
import torch.nn as nn


class ILoss(nn.Module):

    @property
    def weight(self):
        return self._weight

    @property
    def name(self):
        return self._name

    @property
    def reduction(self):
        return "mean"  # Hard-coded average reduction.

    def __init__(self, weight=1.0, name=None):
        super(ILoss, self).__init__()

        assert isinstance(weight, float) and weight >= 0
        self._weight = weight

        name = name or "loss"
        assert isinstance(name, str)
        self._name = name

    @abc.abstractmethod
    def _fn(self, y_pred, y, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, y_pred, y, **kwargs):
        return self._fn(y_pred, y, **kwargs)
