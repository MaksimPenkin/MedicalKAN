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
            self.backbone = nn.Sequential(
                SpectralConv2d(hid_ch, hid_ch, **kwargs),
                SpectralConv2d(hid_ch, hid_ch, **kwargs),
                SpectralConv2d(hid_ch, hid_ch, **kwargs)
            )
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")

        self.restore = nn.Linear(hid_ch, out_ch)

    def forward(self, x):
        x = self.emb(x)
        x = self.backbone(x)
        x = self.restore(x)

        return x
