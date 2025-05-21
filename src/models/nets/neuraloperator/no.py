# """
# @author   Maksim Penkin
# """

from torch import nn
# from .hartley import HartleyLayer
from .fourier import SpectralConv2d


class NeuralOperator(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=64, version="fourier", **kwargs):
        super(NeuralOperator, self).__init__()

        self.emb = nn.Linear(in_ch, hid_ch)

        if version == "fourier":
            self.backbone = nn.ModuleList([SpectralConv2d(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")

        self.restore = nn.Linear(hid_ch, out_ch)
        self.activation = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x = self.emb(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.backbone:
            x = self.activation(layer(x))

        x = x.permute(0, 2, 3, 1)
        x = self.restore(x)
        x = x.permute(0, 3, 1, 2)

        return x
