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
            KANLinear(input_dim, output_dim, spline_order=degree),
            ChebyKANLinear(input_dim, output_dim, degree),
            HermiteFuncKANLinear(input_dim, output_dim, degree)
        ])
        self.mha = nn.MultiheadAttention(output_dim * len(self.subspaces), num_heads=1, batch_first=True)
        self.proj = nn.Linear(output_dim * len(self.subspaces), output_dim)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))

        x = torch.cat([layer(x) for layer in self.subspaces], dim=-1)
        x, _ = self.mha(x, x, x)
        x = self.proj(x)

        return x.view(-1, self.outdim)
