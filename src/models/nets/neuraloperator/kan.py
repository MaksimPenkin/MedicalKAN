# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn

from ..srkan.linear import KANLinear
from ..srkan.chebyshev import ChebyKANLinear
from ..srkan.hermite import HermiteFuncKANLinear


class KAN(nn.Module):

    def __init__(self, in_channels, out_channels, poly=None, **kwargs):
        super(KAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if not poly:
            self.fc = KANLinear(in_channels, out_channels, **kwargs)
        elif poly == "cheby":
            self.fc = ChebyKANLinear(in_channels, out_channels, **kwargs)
        elif poly == "hermite":
            self.fc = HermiteFuncKANLinear(in_channels, out_channels, **kwargs)

        self.norm = nn.LayerNorm(in_channels)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Pre-process.
        B, C, H, W = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = torch.reshape(x, (-1, self.in_channels))

        # Apply KAN.
        y = self.fc(self.norm(x))

        # Post-process.
        y = y.view(B, H * W, self.out_channels).contiguous()
        return y.transpose(1, 2).view(B, self.out_channels, H, W)
