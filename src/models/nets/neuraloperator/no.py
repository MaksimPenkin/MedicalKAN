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

from ..layers import conv1x1, conv3x3, conv5x5, activate, ResBlock


class NeuralOperator(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=32, backbone="fourier", activation="relu", **kwargs):
        super(NeuralOperator, self).__init__()

        self.embedding = conv5x5(in_ch, hid_ch // 2)
        self.lifting = conv3x3(hid_ch // 2, hid_ch)
        if backbone == "fourier2d":
            self.backbone = nn.ModuleList([FourierSpectralConv2d(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        elif backbone == "hartley2d":
            self.backbone = nn.ModuleList([HartleySpectralConv2d(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        elif backbone == "mlp":
            self.backbone = nn.ModuleList([MLP(hid_ch, hid_ch, activation=activation, **kwargs) for _ in range(3)])
        elif backbone == "kan":
            self.backbone = nn.ModuleList([KAN(hid_ch, hid_ch, **kwargs) for _ in range(3)])
        elif backbone == "funkan":
            self.backbone = nn.ModuleList([FUNKAN(hid_ch, hid_ch, **kwargs) for _ in range(3)])
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



# class ConvEncoderBlock(nn.Module):
#
#     def __init__(self, in_ch, out_ch):
#         super(ConvEncoderBlock, self).__init__()
#
#         self.feat = ResBlock(in_ch)
#         self.proj = conv3x3(in_ch, out_ch, stride=2)
#
#     def forward(self, x):
#         feat = self.feat(x)
#         x = self.proj(feat)
#
#         return x, feat
#
#
# class ConvDecoderBlock(nn.Module):
#
#     def __init__(self, in_ch, out_ch):
#         super(ConvDecoderBlock, self).__init__()
#
#         self.up = nn.Upsample(scale_factor=2)
#         self.proj = conv3x3(in_ch, out_ch)
#         self.feat = ResBlock(out_ch)
#
#     def forward(self, x, skip):
#         x = self.proj(self.up(x))
#         x = self.feat(x)
#
#         return x + skip
#
#
# class NeuralOperator2(nn.Module):
#
#     def __init__(self, *args, **kwargs):
#         super(NeuralOperator2, self).__init__()
#
#         filters = 8
#
#         self.emb = conv3x3(3, filters)
#         self.encoder = nn.ModuleList([])
#         for i in range(3):
#             self.encoder.append(
#                 ConvEncoderBlock(filters, filters * 2)
#             )
#             filters = filters * 2
#
#         self.bottleneck = nn.ModuleList([FUNKAN(filters, filters, r=6) for _ in range(3)])
#
#         self.decoder = nn.ModuleList([])
#         for i in range(3):
#             self.decoder.append(
#                 ConvDecoderBlock(filters, filters // 2)
#             )
#             filters = filters // 2
#
#         self.restore = conv3x3(filters, 1)
#
#     def forward(self, x):
#         # Space -> Feature.
#         x = self.emb(x)
#
#         # Encoder.
#         skips = {}
#         i = 0
#         for layer in self.encoder:
#             x, skip = layer(x)
#             i += 1
#             skips[f"enc-{i}"] = skip
#
#         # Bottleneck.
#         for layer in self.bottleneck:
#             x = layer(x)
#
#         # Decoder.
#         for j, layer in enumerate(self.decoder):
#             x = layer(x, skips[f"enc-{i - j}"])
#
#         # Feature -> Space.
#         x = self.restore(F.relu(x))
#
#         return x
