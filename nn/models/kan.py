# """
# @author   https://github.com/AntonioTepsich/Convolutional-KANs
# """

import torch
from torch import nn
from nn.layers.kan_convolutional.KANConv import KAN_Convolutional_Layer


class KAN_v0(nn.Module):
    def __init__(self, n_convs=16, device="cpu"):
        super().__init__()

        self.conv_kan = KAN_Convolutional_Layer(
            n_convs=n_convs,
            kernel_size=(3, 3),
            padding=(1, 1),
            device=device
        )

        self.restore = torch.nn.Conv2d(n_convs, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv_kan(x)
        x = self.restore(x)

        return x
