# """
# @author   Maksim Penkin
# """

import torch.nn as nn
from ..layers import activate


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, activation="relu", **kwargs):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(in_channels, out_channels, **kwargs)
        self.activation = activate(activation)

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

        x = self.fc(x).view(B, H * W, self.out_channels).contiguous()
        x = x.transpose(1, 2).view(B, self.out_channels, H, W)

        return self.activation(x)
