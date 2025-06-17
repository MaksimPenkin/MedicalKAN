# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn

from ..layers import activate


def hermite_funcs(N, r=6):
    x = torch.tanh(torch.linspace(-1, 1, N)) * (math.sqrt(2 * r + 1) + 1)

    psi = torch.ones(r, N, device=x.device)
    psi[0, :] = math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
    if r > 0:
        psi[1, :] = math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
    for k in range(2, r):
        psi[k, :] = math.sqrt(2 / k) * x * psi[k - 1, :].clone() - math.sqrt((k - 1) / k) * psi[k - 2, :].clone()

    return psi


class BasisFuncs(nn.Module):

    def __init__(self, r=6, version="hermite"):
        super(BasisFuncs, self).__init__()

        self.r = r
        self.version = version

    def forward(self, x):
        if self.version == "hermite":
            return hermite_funcs(x.shape[-1], self.r)
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {self.version}.")


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.basis = BasisFuncs(**kwargs)
        self.fc = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.activation = activate(activation)

        self.norm = nn.LayerNorm(in_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=2)

        x = self.norm(x)  # (B, n, N): Get n feature functions, discretized by N nodes.
        psi = self.basis(x)  # (B, r, N): Get r basis functions, discretized by N nodes.
        cost = torch.bmm(x, psi.transpose(1, 2))  # (B, n, r): Calculate cost matrix of feature functions projections on basis functions.
        x = torch.bmm(cost, psi)  # (B, n, N): Construct Kolmogorov-Arnold functions as decomposition over basis functions (with attention).

        x = self.fc(x).view(B, self.out_channels , H * W).contiguous()  # Convolve over all feature functions like in K-A theorem.

        return self.activation(x)
