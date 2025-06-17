# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn


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

    def __init__(self, in_channels, out_channels, poly=None, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if not poly:
            self.basis = RandomFuncs(in_channels, **kwargs)
        elif poly == "hermite":
            self.basis = HermiteFuncs(in_channels, **kwargs)
        else:
            raise NotImplementedError(f"Unrecognized `poly` found: {poly}.")

        self.fc = nn.Conv1d(in_channels, out_channels, 1, bias=False)

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

        psi = self.basis(x)
        cost = torch.bmm(x, psi.transpose(1, 2))
        x = torch.bmm(cost, psi)

        x = self.fc(x.transpose(1, 2)).view(B, self.out_channels, H * W).contiguous()
        x = x.view(B, self.out_channels, H, W)

        return x
