# """
# @author   Maksim Penkin
# """

from torch import nn
from torch.nn import functional as F

from .fourier import FourierSpectralConv2d
from .hartley import HartleySpectralConv2d
from .mlp import MLP
from .kan import KAN
from .funkan import FUNKAN

from ..layers import conv1x1, conv5x5, ConvBlock, ResidualEncoderBlock, ResidualDecoderBlock


def _make_backbone(layer, *args, **kwargs):
    if layer == "fourier2d":
        return FourierSpectralConv2d(*args, **kwargs)
    elif layer == "hartley2d":
        return HartleySpectralConv2d(*args, **kwargs)
    elif layer == "mlp":
        return MLP(*args, **kwargs)
    elif layer == "kan":
        return KAN(*args, **kwargs)
    elif layer == "funkan":
        return FUNKAN(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized `layer` found: {layer}.")


class NeuralOperator(nn.Module):

    def __init__(self, in_ch, out_ch=None, filters=(16, 32), backbone="fourier2d", depth=3, lifting=None, projection=None, skip=False, **kwargs):
        super(NeuralOperator, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        assert len(filters) - 1 >= 1, "at least 2 filters are required"
        self.skip = bool(skip)

        # Embedding.
        self.embedding = conv5x5(in_ch, filters[0])

        # 1. Lifting.
        if lifting == "u-enc":
            self.lifting = nn.ModuleList([ResidualEncoderBlock(filters[i], filters[i + 1]) for i in range(len(filters) - 1)])
        else:
            self.lifting = nn.ModuleList([ConvBlock(filters[i], filters[i + 1], layer="conv3x3") for i in range(len(filters) - 1)])

        # 2. Backbone.
        self.backbone = nn.ModuleList([_make_backbone(backbone, filters[-1], filters[-1], **kwargs) for _ in range(depth)])

        # 3. Projection.
        if projection == "u-dec":
            self.projection = nn.ModuleList([ResidualDecoderBlock(filters[i], filters[i - 1]) for i in range(len(filters) - 1, 0, -1)])
        else:
            self.projection = nn.ModuleList([ConvBlock(filters[i], filters[i - 1], layer="conv1x1") for i in range(len(filters) - 1, 0, -1)])

        # Restoration.
        self.restoration = conv1x1(filters[0], out_ch)

    def forward(self, x):
        # Embedding.
        x = self.embedding(x)
        # 1. Lifting.
        feats = {}
        for i, layer in enumerate(self.lifting):
            if self.skip:
                x, feat = layer(x, return_feature=True)
                feats[f"enc-{i}"] = feat
            else:
                x = layer(x)
        # 2. Backbone.
        for layer in self.backbone:
            x = x + layer(x)
        # 3. Projection.
        for j, layer in enumerate(self.projection):
            if self.skip:
                x = layer(x, feats[f"enc-{i - j}"])
            else:
                x = layer(x)
        # Restoration.
        x = self.restoration(F.relu(x))
        return x
