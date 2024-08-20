# """
# @author   https://github.com/CUHK-AIM-Group/U-KAN
# @author   Maksim Penkin
# """

import math

import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import to_2tuple, trunc_normal_
from nn.layers.kan_original.KANLinear import KANLinear


class PatchEmbedding(nn.Module):

    def __init__(self, in_ch, embed_ch, patch_size=7, stride=4):
        super(PatchEmbedding, self).__init__()

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


class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(x))

        return identity + x


class UKAN(nn.Module):

    def __init__(self, filters=8, L=1, kan_filters=None, K=1, version="spline"):
        super(UKAN, self).__init__()

        filter_list = [filters * (2 ** i) for i in range(L)]
        kan_filters = kan_filters or filter_list[-1] * 2

        self.emb = nn.Conv2d(1, filters, 3, padding=1)
        self.encoder = []
        for i in range(L - 1):
            in_ch, out_ch = filter_list[i], filter_list[i + 1]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    ResBlock(out_ch))
            )

        self.bottleneck_emb = PatchEmbedding(filter_list[-1], kan_filters, 3)
        self.bottleneck = []
        for i in range(K):
            self.bottleneck.append(
                KANBottleneckBlock(kan_filters, version=version)
            )

        self.decoder = []
        for i in range(L - 1):
            in_ch, out_ch = filter_list[i + 1], filter_list[i]
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    ResBlock(out_ch))
            )
        self.restore = nn.Conv2d(filter_list[0], 1, 3, padding=1)

    def forward(self, x):
        # Space -> Feature.
        x = self.emb(x)

        # Encoder.
        skips = {
            "enc-1": x
        }
        for idx, layer in enumerate(self.encoder, 1):
            x = layer(x)
            skips[f"enc-{idx + 1}"] = x

        # Bottleneck.
        x, H, W = self.bottleneck_emb(x)
        _, _, C = x.shape
        for layer in enumerate(self.bottleneck):
            x = layer(x, H, W)
        x = self.norm(x)
        x = x.reshape(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Decoder.
        for idx, layer in reversed(list(enumerate(self.decoder))):
            x = layer(x)
            x += skips[f"enc-{idx + 1}"]

        # Feature -> Space.
        x = self.restore(F.relu(x))

        return x
