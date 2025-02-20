# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn

from nn.layers.kan_original.KANLinear import KANLinear
from nn.layers.kan_advanced.chebyshev import ChebyKANLinear
from nn.layers.kan_advanced.hermite import HermiteFuncKANLinear


class AttentionKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(AttentionKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.subspaces = nn.ModuleList([
            # KANLinear(input_dim, output_dim, spline_order=degree),
            ChebyKANLinear(input_dim, output_dim, degree, einsum=False),
            HermiteFuncKANLinear(input_dim, output_dim, degree, einsum=False)
        ])
        # self.mha = nn.MultiheadAttention(output_dim * len(self.subspaces), num_heads=1, batch_first=True)
        # self.proj = nn.Linear(output_dim * len(self.subspaces), output_dim)

        self.mha = nn.MultiheadAttention(output_dim, num_heads=1, batch_first=True)
        self.coeffs = nn.Parameter(torch.empty(output_dim, output_dim, (degree + 1) * len(self.subspaces)))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (output_dim * (degree + 1) * len(self.subspaces)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))

        # x = torch.cat([layer(x) for layer in self.subspaces], dim=-1)
        # x, _ = self.mha(x, x, x)
        # x = self.proj(x)
        # return x.view(-1, self.outdim)

        x = torch.cat([layer(x) for layer in self.subspaces], dim=-1).permute(0, 2, 1)
        x = self.mha(x, x, x)[0].permute(0, 2, 1)
        x = torch.einsum('bid,iod->bo', x, self.coeffs)
        return x
