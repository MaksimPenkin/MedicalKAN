# """
# @author   Maksim Penkin
# """

from torch import nn
from ..layers import conv3x3, ResBlock


class ResidualEncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, **kwargs):
        super(ResidualEncoderBlock, self).__init__()

        self.feat = ResBlock(in_ch, **kwargs)
        self.down = conv3x3(in_ch, out_ch, stride=2)

    def forward(self, x):
        feat = self.feat(x)
        x = self.down(feat)
        return x, feat


class ResidualEncoder(nn.Module):

    def __init__(self, filters, **kwargs):
        super(ResidualEncoder, self).__init__()

        num_blocks = len(filters) - 1
        assert num_blocks >= 1

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.blocks.append(
                ResidualEncoderBlock(filters[i], filters[i + 1], **kwargs)
            )

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x, feat = block(x)
            feats.append(feat)
        return x, feats
