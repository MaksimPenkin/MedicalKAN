# """
# @author   Maksim Penkin
# """

from torch import nn
from ..layers import ResidualEncoderBlock


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
