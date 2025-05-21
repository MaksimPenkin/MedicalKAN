# """
# @author   Maksim Penkin
# """

import math
import torch
import torch.nn as nn


class Hermite2d(nn.Module):

    def __init__(self, in_channels, out_channels, degree, eps=1):
        super(Hermite2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.eps = eps

        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, degree + 1))
        nn.init.normal_(self.weights, mean=0.0, std=1 / (in_channels * (degree + 1)))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = torch.reshape(x, (-1, self.in_channels))

        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x) * (math.sqrt(2 * self.degree + 1) + self.eps)
        hermite = torch.ones(x.shape[0], self.in_channels, self.degree + 1, device=x.device)
        hermite[:, :, 0] = math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
        if self.degree > 0:
            hermite[:, :, 1] = math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = math.sqrt(2 / i) * x * hermite[:, :, i - 1].clone() - math.sqrt((i - 1) / i) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.weights).view(-1, self.out_channels)

        y = y.view(B, H * W, self.out_channels).contiguous()
        return y.transpose(1, 2).view(B, self.out_channels, H, W)
