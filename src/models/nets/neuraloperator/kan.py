# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn

from ..srkan.linear import KANLinear
from ..srkan.chebyshev import ChebyKANLinear
from ..srkan.hermite import HermiteFuncKANLinear


class KAN(nn.Module):

    def __init__(self, dim, poly=None, **kwargs):
        super(KAN, self).__init__()

        if not poly:
            self.fc = KANLinear(dim, dim, **kwargs)
        elif poly == "cheby":
            self.fc = ChebyKANLinear(dim, dim, **kwargs)
        elif poly == "hermite":
            self.fc = HermiteFuncKANLinear(dim, dim, **kwargs)

        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.fc(self.norm(x)).view(B, H * W, C).contiguous()
        x = x.transpose(1, 2).view(B, C, H, W)

        return x + identity
