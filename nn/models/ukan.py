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

    def __init__(self, dim, version="spline", grid_size=5, spline_order=3):
        super(BottleneckBlock, self).__init__()

        if version == "spline":
            self.fc = KANLinear(dim, dim,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                scale_noise=0.1,
                                scale_base=1.0,
                                scale_spline=1.0,
                                base_activation=torch.nn.SiLU,
                                grid_eps=0.02,
                                grid_range=[-1, 1])
        elif version == "cheby":
            self.fc = ChebyKANLinear(dim, dim, spline_order)
        elif version == "linear":
            self.fc = nn.Linear(dim, dim)
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

        if self.fc is not None:
            identity = x

            x = x.reshape(B * N, C)
            x = self.fc(x)
            x = x.reshape(B, N, C).contiguous()

            return self.norm(identity + x)
        else:
            return x


class StackedResidualKAN(nn.Module):

    def __init__(self, filters=8, S=1, L=3, **kwargs):
        super(StackedResidualKAN, self).__init__()
        assert S >= 1 and L >= 1

        self.emb = conv3x3(1, filters)
        self.encoder = nn.ModuleList([])
        for i in range(S):
            self.encoder.append(
                ConvEncoderBlock(filters, filters * 2)
            )
            filters = filters * 2

        self.bottleneck_enc = PatchEncoder(filters, 16, patch_size=5)
        self.bottleneck = nn.ModuleList([])
        for i in range(L):
            self.bottleneck.append(
                BottleneckBlock(16, **kwargs)
            )
        self.bottleneck_dec = PatchDecoder(16, filters, patch_size=5)

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
