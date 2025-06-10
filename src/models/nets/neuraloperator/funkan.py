# """
# @author   Maksim Penkin
# """

import torch.nn as nn

from ..srkan.linear import KANLinear
from ..srkan.chebyshev import ChebyKANLinear
from ..srkan.hermite import HermiteFuncKANLinear


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = nn.LayerNorm(in_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        return x
