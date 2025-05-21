# """
# @author   https://github.com/SynodicMonth/ChebyKAN
# """

import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, einsum=True):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.einsum = bool(einsum)
        if self.einsum:
            self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()

        if self.einsum:
        # Compute the Chebyshev interpolation
            y = torch.einsum(
                "bid,iod->bo", x, self.cheby_coeffs
            )  # shape = (batch_size, outdim)
            return y.view(-1, self.outdim)
        else:
            return x
