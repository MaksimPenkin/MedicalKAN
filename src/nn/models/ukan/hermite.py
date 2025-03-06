# """
# @author   https://github.com/Boris-73-TA/OrthogPolyKANs
# @author   https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/hermite_kan.py
# """

import math
import torch
import torch.nn as nn


class HermiteKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, einsum=True):
        super(HermiteKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.einsum = bool(einsum)
        if self.einsum:
            self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
            nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        hermite = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i - 1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()

        if self.einsum:
            y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
            return y.view(-1, self.outdim)
        else:
            return hermite


class HermiteFuncKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, eps=1, einsum=True):
        super(HermiteFuncKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.eps = eps

        self.einsum = bool(einsum)
        if self.einsum:
            self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
            nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x) * (math.sqrt(2 * self.degree + 1) + self.eps)
        hermite = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        hermite[:, :, 0] = math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
        if self.degree > 0:
            hermite[:, :, 1] = math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = math.sqrt(2 / i) * x * hermite[:, :, i - 1].clone() - math.sqrt((i - 1) / i) * hermite[:, :, i - 2].clone()

        if self.einsum:
            y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
            return y.view(-1, self.outdim)
        else:
            return hermite
