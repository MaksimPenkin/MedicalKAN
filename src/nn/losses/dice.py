# """
# @author   IvanDrokin https://github.com/IvanDrokin/torch-conv-kan/blob/main/train/losses.py
# """

import torch
from torch import nn
from torch.nn import functional as F


class DiceWithLogits(nn.Module):

    def __init__(self, smooth=1.0):
        super(DiceWithLogits, self).__init__()

        self.smooth = smooth

    def _channel_with_dice(self, inputs, targets):
        dice = 0.
        for i in range(inputs.shape[1]):
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


class DiceWithLogitsLoss(DiceWithLogits):
    def __init__(self, *args, **kwargs):
        super(DiceWithLogitsLoss, self).__init__(*args, **kwargs)

    def forward(self, inputs, targets):
        return 1. - super(DiceWithLogitsLoss, self).forward(inputs, targets)
