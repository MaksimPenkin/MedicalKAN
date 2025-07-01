# """
# @author   Maksim Penkin
# """

import math

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_

from .linear import KANLinear
from .chebyshev import ChebyKANLinear
from .hermite import HermiteKANLinear, HermiteFuncKANLinear
from .attention import AttentionKANLinear

from ..layers import ResBlock, conv3x3, conv1x1


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    n, c, h, w = x.size()
    return F.pixel_shuffle(x, block_size)


class PatchEncoder(nn.Module):

    @property
    def patch_size(self):
        return self._patch_size

    def __init__(self, in_ch, out_ch, patch_size=7):
        super(PatchEncoder, self).__init__()

        self._patch_size = patch_size
        self.proj = conv1x1(in_ch * self.patch_size * self.patch_size, out_ch)

    def forward(self, x):
        x = space_to_depth(x, self.patch_size)
        x = self.proj(x)

        return x


class PatchDecoder(nn.Module):

    @property
    def patch_size(self):
        return self._patch_size

    def __init__(self, in_ch, out_ch, patch_size=7):
        super(PatchDecoder, self).__init__()

        self._patch_size = patch_size
        self.proj = conv1x1(in_ch, out_ch * self.patch_size * self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = depth_to_space(x, self.patch_size)

        return x


class ConvEncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvEncoderBlock, self).__init__()

        self.feat = ResBlock(in_ch)
        self.proj = conv3x3(in_ch, out_ch)

    def forward(self, x):
        feat = self.feat(x)
        x = self.proj(feat)

        return x, feat


class ConvDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvDecoderBlock, self).__init__()

        self.proj = conv3x3(in_ch, out_ch)
        self.feat = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.proj(x)
        x = self.feat(x)

        return x + skip


class BottleneckBlock(nn.Module):

    def __init__(self, dim, version="spline", **kwargs):
        super(BottleneckBlock, self).__init__()

        if version == "spline":
            self.fc = KANLinear(dim, dim, **kwargs)
        elif version == "cheby":
            self.fc = ChebyKANLinear(dim, dim, **kwargs)
        elif version == "hermite_poly":
            self.fc = HermiteKANLinear(dim, dim, **kwargs)
        elif version == "hermite_func":
            self.fc = HermiteFuncKANLinear(dim, dim, **kwargs)
        elif version == "linear":
            self.fc = nn.Linear(dim, dim)
        elif version == "mha":
            self.fc = AttentionKANLinear(dim, dim, **kwargs)
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
        B, C, H, W = x.shape
        identity = x

        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.fc(self.norm(x)).view(B, H * W, C).contiguous()
        x = x.transpose(1, 2).view(B, C, H, W)

        return x + identity


class StackedResidualKAN(nn.Module):

    def __init__(self, filters=4, S=1, L=3, **kwargs):
        super(StackedResidualKAN, self).__init__()
        assert S >= 1 and L >= 1

        self.emb = conv3x3(1, filters)
        self.encoder = nn.ModuleList([])
        for i in range(S):
            self.encoder.append(
                ConvEncoderBlock(filters, filters * 2)
            )
            filters = filters * 2

        self.bottleneck_enc = PatchEncoder(filters, 64, patch_size=5)
        self.bottleneck = nn.ModuleList([])
        for i in range(L):
            self.bottleneck.append(
                BottleneckBlock(64, **kwargs)
            )
        self.bottleneck_dec = PatchDecoder(64, filters, patch_size=5)

        self.decoder = nn.ModuleList([])
        for i in range(S, 0, -1):
            self.decoder.append(
                ConvDecoderBlock(filters, filters // 2)
            )
            filters = filters // 2

        self.restore = conv3x3(filters, 1)

    def forward(self, x):
        # Space -> Feature.
        x = self.emb(x)

        # Encoder.
        skips = {}
        i = 0
        for layer in self.encoder:
            x, skip = layer(x)
            i += 1
            skips[f"enc-{i}"] = skip

        # Bottleneck.
        x = self.bottleneck_enc(x)
        for layer in self.bottleneck:
            x = layer(x)
        x = self.bottleneck_dec(x)

        # Decoder.
        for j, layer in enumerate(self.decoder):
            x = layer(x, skips[f"enc-{i - j}"])

        # Feature -> Space.
        x = self.restore(F.relu(x))

        return x
