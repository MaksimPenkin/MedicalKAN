# """
# @author   https://github.com/Boris-73-TA/OrthogPolyKANs
# @author   https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/hermite_kan.py
# """

import math
import torch
import torch.nn as nn


class HermiteKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HermiteKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree

        # Initialize Hermite polynomial coefficients
        self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        hermite = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i - 1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(-1, self.out_dim)
        return y


class HermiteFuncKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, eps=1):
        super(HermiteFuncKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.eps = eps

        self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x) * (math.sqrt(2 * self.degree + 1) + self.eps)
        hermite = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        hermite[:, :, 0] = math.pi ** (-1 / 4) * torch.exp(-(x ** 2) / 2)
        if self.degree > 0:
            hermite[:, :, 1] = math.sqrt(2) * math.pi ** (-1 / 4) * x * torch.exp(-(x ** 2) / 2)
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = math.sqrt(2 / i) * x * hermite[:, :, i - 1].clone() - math.sqrt((i - 1) / i) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(-1, self.out_dim)
        return y
