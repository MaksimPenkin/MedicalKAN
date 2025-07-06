# """
# @author   Maksim Penkin
# """

from torch import nn
from ..layers import ResidualDecoderBlock


class ResidualDecoder(nn.Module):

    def __init__(self, filters, **kwargs):
        super(ResidualDecoder, self).__init__()

        assert len(filters) - 1 >= 1, "at least 2 filters are required"
        self.blocks = nn.ModuleList([ResidualDecoderBlock(filters[i], filters[i + 1], **kwargs) for i in range(len(filters) - 1)])

    def forward(self, x, feats):
        j = len(feats) - 1
        for i, block in enumerate(self.blocks):
            x = block(x, feats[j - i])
        return x
