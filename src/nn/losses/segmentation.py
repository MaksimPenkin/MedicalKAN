import torch
from torch import nn
from torch.nn import functional as F


class Dice(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(Dice, self).__init__()
        self.smooth = smooth

    def _channel_with_dice(self, inputs, targets):
        dice = 0.
        for i in range(inputs.shape[1]):
            # flatten label and prediction tensors

            c_inp = inputs[:, i].view(inputs.shape[0], -1)
            c_tgt = targets[:, i].view(inputs.shape[0], -1)

            intersection = torch.sum(c_inp * c_tgt, dim=1)
            dice += (2. * intersection + self.smooth) / (c_inp.sum(dim=1) + c_tgt.sum(dim=1) + self.smooth)
        dice = torch.mean(dice) / float(inputs.shape[1])
        return dice

    def forward(self, inputs, targets):
        if inputs.shape[1] > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.sigmoid(inputs)
        dice = self._channel_with_dice(inputs, targets)
        return dice


class DiceLoss(Dice):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__(smooth=smooth)

    def forward(self, inputs, targets):
        return 1. - super(DiceLoss, self).forward(inputs, targets)


class DiceLossWithBCE(DiceLoss):
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super(DiceLossWithBCE, self).__init__(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer

        dice = super(DiceLossWithBCE, self).forward(inputs, targets)
        inputs = F.sigmoid(inputs)

        dice = self.bce_weight * F.binary_cross_entropy(inputs, targets) + dice
        return dice
