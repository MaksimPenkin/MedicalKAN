# """
# @author   Maksim Penkin
# """

import torch
import torch.nn.functional as F
from torch import nn


class HNO(nn.Module):

    def __init__(self, *args, **kwargs):
        super(HNO, self).__init__()

    def forward(self, x):
        return x
