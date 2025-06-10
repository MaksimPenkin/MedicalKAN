# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HermiteFuncs(nn.Module):
    # https://github.com/Rob217/hermite-functions/blob/master/hermite_functions/hermite_functions.py
    def __init__(self, r, n):
        super(HermiteFuncs, self).__init__()
        self.r = r
        self.register_buffer("x_d", torch.linspace(-1, 1, n))

    def forward(self, x):
        B, HW, C = x.shape

        x_d = torch.tanh(self.x_d) * (math.sqrt(2 * self.r + 1) + 1)

        psi = torch.ones(self.r + 1, C, device=x.device)
        psi[0, :] = math.pi ** (-1 / 4) * torch.exp(-(x_d ** 2) / 2)
        if self.r > 0:
            psi[1, :] = math.sqrt(2) * math.pi ** (-1 / 4) * x_d * torch.exp(-(x_d ** 2) / 2)
        for k in range(2, self.r + 1):
            psi[k, :] = math.sqrt(2 / k) * x_d * psi[k - 1, :].clone() - math.sqrt((k - 1) / k) * psi[k - 2, :].clone()

        return psi.unsqueeze(0).repeat(B, 1, 1)


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, degree=5, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.basis = HermiteFuncs(degree, in_channels)
        self.fc = nn.Conv1d(in_channels, out_channels, 1)

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

        q_x = x
        x = self.basis(x)
        scores = torch.bmm(q_x, x.transpose(1,2)) / (self.in_channels ** 0.5)
        x = torch.bmm(F.softmax(scores, dim=2), x)

        x = self.fc(x.transpose(1, 2)).view(B, self.out_channels, H*W).contiguous()

        return x.view(B, self.out_channels, H, W)
