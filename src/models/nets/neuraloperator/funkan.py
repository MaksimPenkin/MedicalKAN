# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hermite_function(n, x):
    if n == 0:
        return math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
    elif n == 1:
        return math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
    else:
        return math.sqrt(2 / n) * x * hermite_function(n - 1, x) - math.sqrt((n - 1) / n) * hermite_function(n - 2, x)


class Hermite2d(nn.Module):

    def __init__(self, r=6):
        super(Hermite2d, self).__init__()

        self.r = r
        self.fc1 = nn.Linear(1, 8, bias=False)
        self.fc2 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        B, n, H, W = x.shape

        x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                        torch.linspace(-1, 1, W, device=x.device),
                                        indexing='xy')

        x_field, y_field = torch.split(self.fc2(F.relu(self.fc1(x.unsqueeze(-1)))), 1, dim=-1)

        x_grid = x_field.squeeze(-1) + x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = y_field.squeeze(-1) + y_grid.unsqueeze(0).unsqueeze(0)
        return torch.stack([hermite_function(k, x_grid) * hermite_function(k, y_grid) for k in range(self.r)], 2)


class FUNKAN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FUNKAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Kolmogorov-Arnold functions.
        self.phi = nn.Parameter(torch.empty(in_channels, 145, 145))
        nn.init.orthogonal_(self.phi)
        # Basis functions.
        self.psi = Hermite2d(**kwargs)
        # Theta params.
        self.theta = nn.Conv2d(in_channels, out_channels, 1)

        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        psi = self.psi(self.norm(x))  # (B, n, r, H, W): basis Hermite functions.
        cost = torch.einsum("bikd,id->bik", psi.flatten(start_dim=3), self.phi.flatten(start_dim=1))  # (B, n, r)
        x = torch.einsum("bik,bikd->bid", F.softmax(cost, dim=2), psi.flatten(start_dim=3)).view(B, self.in_channels, H, W)
        x = self.theta(x)

        return x
