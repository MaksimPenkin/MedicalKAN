# """
# @author   https://github.com/CUHK-AIM-Group/U-KAN
# @author   Maksim Penkin
# """

import math

import torch
from torch import nn
from timm.models.layers import to_2tuple, trunc_normal_

from nn.layers.kan_original.KANLinear import KANLinear


class _PatchEmbedding(nn.Module):

    def __init__(self, in_ch, embed_ch, patch_size=7, stride=4):
        super(_PatchEmbedding, self).__init__()

        patch_size = to_2tuple(patch_size)
        self.emb = nn.Conv2d(in_ch, embed_ch, patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

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
        x = self.emb(x)
        _, _, H, W = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)

        return x, H, W


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
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")

        self.norm = nn.LayerNorm(dim)

        self.dwconv = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

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

    def forward(self, x, H, W):
        B, N, C = x.shape
        identity = x

        x = x.reshape(B * N, C)
        x = self.kan(self.norm(x))
        x = x.reshape(B, N, C).contiguous()

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        x = x.flatten(start_dim=2).transpose(1, 2)

        return identity + x


class ConvEncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvEncoderBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvDecoderBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
