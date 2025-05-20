# """
# @author   Maksim Penkin
# """

from torch import nn
from ..layers import conv3x3, ResBlock


class ResidualDecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, **kwargs):
        super(ResidualDecoderBlock, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            conv3x3(in_ch, out_ch)
        )
        self.feat = ResBlock(out_ch, **kwargs)

    def forward(self, x, skip):
        x = self.up(x) + skip
        x = self.feat(x)
        return x


class ResidualDecoder(nn.Module):

    def __init__(self, filters, **kwargs):
        super(ResidualDecoder, self).__init__()

        num_blocks = len(filters) - 1
        assert num_blocks >= 1

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.blocks.append(
                ResidualDecoderBlock(filters[i], filters[i + 1], **kwargs)
            )

    def forward(self, x, feats):
        j = len(feats) - 1
        for i, block in enumerate(self.blocks):
            x = block(x, feats[j - i])
        return x
