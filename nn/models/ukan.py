# """
# @author   https://github.com/CUHK-AIM-Group/U-KAN
# @author   Maksim Penkin
# """

import math

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_

from nn.models.resnet import ResBlock, conv3x3, conv1x1
from nn.layers.kan_original.KANLinear import KANLinear
from nn.layers.kan_advanced.chebyshev import ChebyKANLinear
from nn.transforms.pixel_shuffle import space_to_depth, depth_to_space


class PatchEncoder(nn.Module):

    @property
    def patch_size(self):
        return self._patch_size

    def __init__(self, in_ch, out_ch, patch_size=7):
        super(PatchEncoder, self).__init__()

        self._patch_size = patch_size
        self.proj = conv1x1(in_ch * self.patch_size * self.patch_size, out_ch)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = space_to_depth(x, self.patch_size)
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)

        return self.norm(x), H, W


class PatchDecoder(nn.Module):

    @property
    def patch_size(self):
        return self._patch_size

    def __init__(self, in_ch, out_ch, patch_size=7):
        super(PatchDecoder, self).__init__()

        self._patch_size = patch_size
        self.proj = conv1x1(in_ch, out_ch * self.patch_size * self.patch_size)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(x)
        x = depth_to_space(x, self.patch_size)

        return x


class ConvEncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvEncoderBlock, self).__init__()

        self.proj = conv3x3(in_ch, out_ch)
        self.feat = ResBlock(out_ch)

    def forward(self, x):
        x = self.proj(x)
        x = self.feat(x)

        return x


class ConvDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvDecoderBlock, self).__init__()

        self.feat = ResBlock(in_ch)
        self.proj = conv3x3(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.feat(x + skip)
        x = self.proj(x)

        return x


class KANBottleneckBlock(nn.Module):

    def __init__(self, dim, version="spline"):
        super(KANBottleneckBlock, self).__init__()

        if version == "spline":
            self.kan = KANLinear(dim, dim,
                                 grid_size=5,
                                 spline_order=3,
                                 scale_noise=0.1,
                                 scale_base=1.0,
                                 scale_spline=1.0,
                                 base_activation=torch.nn.SiLU,
                                 grid_eps=0.02,
                                 grid_range=[-1, 1])
        elif version == "cheby":
            self.kan = ChebyKANLinear(dim, dim, 3)
        elif version == "linear":
            self.kan = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            )
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")

        self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        identity = x

        x = x.reshape(B * N, C)
        x = self.kan(x)
        x = x.reshape(B, N, C).contiguous()

        return self.norm(identity + x)


class StackedResidualKAN(nn.Module):

    def __init__(self, filters=8, L=1, kan_filters=None, K=1, version="spline"):
        super(StackedResidualKAN, self).__init__()
        assert L >= 1 and K >= 1

        filter_list = [filters, ] + [filters * (2 ** (i + 1)) for i in range(L)]
        kan_filters = kan_filters or filter_list[-1]

        self.emb = conv3x3(1, filters)
        self.encoder = nn.ModuleList([])
        filters = filter_list[0]
        for i in range(1, L + 1):
            self.encoder.append(
                ConvEncoderBlock(filters, filter_list[i])
            )
            filters = filter_list[i]

        self.bottleneck_enc = PatchEncoder(filters, kan_filters, patch_size=5)
        self.bottleneck = nn.ModuleList([])
        for i in range(K):
            self.bottleneck.append(
                KANBottleneckBlock(kan_filters, version=version)
            )
        self.bottleneck_dec = PatchDecoder(kan_filters, filters, patch_size=5)

        self.decoder = nn.ModuleList([])
        for i in range(L, 0, -1):
            self.decoder.append(
                ConvDecoderBlock(filters, filter_list[i - 1])
            )
            filters = filter_list[i - 1]

        self.restore = conv3x3(filters, 1)

    def forward(self, x):
        # Space -> Feature.
        x = self.emb(x)

        # Encoder.
        skips = {}
        i = 0
        for layer in self.encoder:
            x = layer(x)
            i += 1
            skips[f"enc-{i}"] = x

        # Bottleneck.
        x, H, W = self.bottleneck_enc(x)
        for layer in self.bottleneck:
            x = layer(x)
        x = self.bottleneck_dec(x, H, W)

        # Decoder.
        for j, layer in enumerate(self.decoder):
            x = layer(x, skips[f"enc-{i - j}"])

        # Feature -> Space.
        x = self.restore(F.relu(x))

        return x
