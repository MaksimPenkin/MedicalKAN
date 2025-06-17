# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn

from ..layers import activate


class RandomFuncs(nn.Module):

    def __init__(self, d, r=6, initializer=None):
        super(RandomFuncs, self).__init__()

        self.psi = nn.Parameter(
            torch.empty(
                (r, d)
            )
        )

        if initializer == "orthogonal":
            nn.init.orthogonal_(self.psi)
        else:
            nn.init.kaiming_uniform_(self.psi, a=math.sqrt(5))

    def forward(self, x):
        B, HW, C = x.shape

        return self.psi.unsqueeze(0).repeat(B, 1, 1)


class HermiteFuncs(nn.Module):
    # https://github.com/Rob217/hermite-functions/blob/master/hermite_functions/hermite_functions.py
    def __init__(self, d, r=6):
        super(HermiteFuncs, self).__init__()

        self.r = r
        self.register_buffer("x_d", torch.linspace(-1, 1, d))

    def forward(self, x):
        B, HW, C = x.shape

        x_d = torch.tanh(self.x_d) * (math.sqrt(2 * self.r + 1) + 1)

        psi = torch.ones(self.r, C, device=x.device)
        psi[0, :] = math.pi ** (-1 / 4) * torch.exp(-(x_d ** 2) / 2)
        if self.r > 0:
            psi[1, :] = math.sqrt(2) * math.pi ** (-1 / 4) * x_d * torch.exp(-(x_d ** 2) / 2)
        for k in range(2, self.r):
            psi[k, :] = math.sqrt(2 / k) * x_d * psi[k - 1, :].clone() - math.sqrt((k - 1) / k) * psi[k - 2, :].clone()

        return psi.unsqueeze(0).repeat(B, 1, 1)


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels=None, poly=None, activation=None, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels

        if not poly:
            self.basis = RandomFuncs(in_channels, **kwargs)
        elif poly == "hermite":
            self.basis = HermiteFuncs(in_channels, **kwargs)
        else:
            raise NotImplementedError(f"Unrecognized `poly` found: {poly}.")

        self.fc = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        if out_channels is not None:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.proj = nn.Identity()

        self.activation = activate(activation)

        self.norm = nn.LayerNorm(in_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.norm(x)

        psi = self.basis(x)  # (B, r, d): Get r basis functions, discretized on d nodes.
        cost = torch.bmm(x, psi.transpose(1, 2))  # (B, n, r): Calculate cost matrix of feature functions projections on basis functions.
        x = torch.bmm(cost, psi)  # (B, n, d): Calculate Kolmogorov-Arnold functions as decomposition over basis functions (with attention).

        x = self.fc(x).view(B, H * W, self.in_channels).contiguous()  # Convolve over all feature functions like in K-A theorem.
        x = x.transpose(1, 2).view(B, self.in_channels, H, W)

        return self.proj(self.activation(x))  # Optional functions' (non-linear) re-discretization.
