# """
# @author   Maksim Penkin
# """

from torch import nn

from .fourier import FourierSpectralConv2d
from .hartley import HartleySpectralConv2d
from .kan import KAN
from .funkan import FUNKAN

from ..layers import conv1x1, conv3x3, conv5x5, activate


class NeuralOperator(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=32, backbone="fourier", activation="relu", **kwargs):
        super(NeuralOperator, self).__init__()

        self.embedding = conv5x5(in_ch, hid_ch // 2)
        self.lifting = conv3x3(hid_ch // 2, hid_ch)
        if backbone == "fourier2d":
            self.backbone = nn.ModuleList([FourierSpectralConv2d(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        elif backbone == "hartley2d":
            self.backbone = nn.ModuleList([HartleySpectralConv2d(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        elif backbone == "kan":
            self.backbone = nn.ModuleList([KAN(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        elif backbone == "funkan":
            self.backbone = nn.ModuleList([FUNKAN(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        else:
            raise NotImplementedError(f"Unrecognized `backbone` found: {backbone}.")
        self.projection = conv1x1(hid_ch, hid_ch // 2)
        self.restoration = conv1x1(hid_ch // 2, out_ch)

        self.activation = activate(activation)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lifting(self.activation(x))
        for layer in self.backbone:
            x = x + layer(x)
        x = self.projection(self.activation(x))
        x = self.restoration(self.activation(x))
        return x
