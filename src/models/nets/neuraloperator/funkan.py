# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import activate


class HermiteFuncs(nn.Module):

    def __init__(self, r=6):
        super(HermiteFuncs, self).__init__()

        self.r = r

    def forward(self, x):
        B, n, N = x.shape

        x = torch.tanh(x) * math.sqrt(2 * self.r + 1)
        psi = torch.ones(B, n, self.r, N, device=x.device)
        psi[:, :, 0, :] = math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
        if self.r > 0:
            psi[:, :, 1, :] = math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
        for k in range(2, self.r):
            psi[:, :, k, :] = math.sqrt(2 / k) * x * psi[:, :, k - 1, :].clone() - math.sqrt((k - 1) / k) * psi[:, :, k - 2, :].clone()

        return psi


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.psi = HermiteFuncs(**kwargs)
        self.fc = nn.Conv1d(in_channels, out_channels, 1)
        self.activation = activate(activation)

        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.flatten(start_dim=2)  # (B, n, N): n feature functions.
        x = self.norm(x)

        psi = self.psi(x)  # (B, n, r, N): basis functions.
        cost = torch.einsum("bikd,bid->bik", psi, x)  # (B, n, r)
        x = torch.einsum("bik,bikd->bid", F.softmax(cost, dim=2), psi)  # (B, n, N)
        x = self.fc(x).view(B, self.out_channels, H, W).contiguous()  # Convolve over all feature functions like in K-A theorem.

        return self.activation(x)
