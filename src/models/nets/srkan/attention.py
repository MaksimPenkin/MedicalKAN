# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn

from .linear import KANLinear
from .chebyshev import ChebyKANLinear
from .hermite import HermiteFuncKANLinear


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
        # v1.0
        # self.mha = nn.MultiheadAttention(input_dim * len(self.subspaces), num_heads=1, batch_first=True)
        # self.proj = nn.Linear(input_dim * len(self.subspaces), output_dim)

        # v1.1
        self.mha = nn.MultiheadAttention(input_dim, num_heads=1, batch_first=True)
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, (degree + 1) * len(self.subspaces)))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * (degree + 1) * len(self.subspaces)))

        # v1.2
        # self.mha = nn.MultiheadAttention(input_dim, num_heads=1, batch_first=True)
        # self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))

        # v1.0
        # x = torch.cat([layer(x) for layer in self.subspaces], dim=-1)
        # x, _ = self.mha(x, x, x)
        # x = self.proj(x)
        # return x.view(-1, self.outdim)

        # v1.1
        x = torch.cat([layer(x) for layer in self.subspaces], dim=-1).permute(0, 2, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(0, 2, 1)
        x = torch.einsum('bid,iod->bo', x, self.coeffs)
        return x.view(-1, self.outdim)

        # v1.2
        # q_x = x.unsqueeze(1)
        # k_x = torch.cat([layer(x) for layer in self.subspaces], dim=-1).permute(0, 2, 1)
        # x, m = self.mha(q_x, k_x, k_x)
        # x = self.proj(x.squeeze(1))
        # return x.view(-1, self.outdim)
