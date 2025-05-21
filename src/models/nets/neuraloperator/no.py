# """
# @author   Maksim Penkin
# """

from torch import nn
from .fourier import SpectralConv2d
from .hartley import HartleyConv2d
from .hermite import Hermite2d
from ..layers import conv3x3


class NeuralOperator(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=64, version="fourier", **kwargs):
        super(NeuralOperator, self).__init__()

        self.emb = conv3x3(in_ch, hid_ch)
        if version == "fourier":
            self.backbone = nn.ModuleList([SpectralConv2d(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        elif version == "hartley":
            self.backbone = nn.ModuleList([HartleyConv2d(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        elif version == "hermite":
            self.backbone = nn.ModuleList([Hermite2d(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")
        self.restore = conv3x3(hid_ch, out_ch)

        self.activation = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.emb(x)
        for layer in self.backbone:
            x = self.activation(layer(x) + x)
        x = self.restore(x)

        return x
