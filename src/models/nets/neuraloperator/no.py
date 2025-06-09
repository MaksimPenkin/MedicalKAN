# """
# @author   Maksim Penkin
# """

from torch import nn
import torch.nn.functional as F

from .fourier import FourierSpectralConv2d
from .hartley import HartleySpectralConv2d
from .mlp import MLP
from .kan import KAN

from ..layers import conv1x1, conv3x3, conv5x5


class NeuralOperator(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=32, version="fourier", **kwargs):
        super(NeuralOperator, self).__init__()

        self.embedding = conv5x5(in_ch, hid_ch // 2)
        self.lifting = conv3x3(hid_ch // 2, hid_ch)
        if version == "fourier2d":
            self.backbone = nn.ModuleList([FourierSpectralConv2d(hid_ch, hid_ch, activation="relu", **kwargs) for _ in range(3)])
        elif version == "hartley2d":
            self.backbone = nn.ModuleList([HartleySpectralConv2d(hid_ch, hid_ch, activation="relu", **kwargs) for _ in range(3)])
        elif version == "mlp":
            self.backbone = nn.ModuleList([MLP(hid_ch, hid_ch, activation="sigmoid", **kwargs) for _ in range(3)])
        elif version == "kan":
            self.backbone = nn.ModuleList([KAN(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")
        self.projection = conv1x1(hid_ch, hid_ch // 2)
        self.restoration = conv1x1(hid_ch // 2, out_ch)

    def forward(self, x):
        x = F.relu(self.embedding(x))
        x = self.lifting(x)
        for layer in self.backbone:
            x = x + layer(x)
        x = self.projection(x)
        x = self.restoration(F.relu(x))
        return x
