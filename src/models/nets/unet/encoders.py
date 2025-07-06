# """
# @author   Maksim Penkin
# """

from torch import nn
from ..layers import ResidualEncoderBlock


class ResidualEncoder(nn.Module):

    def __init__(self, filters, **kwargs):
        super(ResidualEncoder, self).__init__()

        assert len(filters) - 1 >= 1, "at least 2 filters are required"
        self.blocks = nn.ModuleList([ResidualEncoderBlock(filters[i], filters[i + 1], **kwargs) for i in range(len(filters) - 1)])

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x, feat = block(x, return_feature=True)
            feats.append(feat)
        return x, feats
