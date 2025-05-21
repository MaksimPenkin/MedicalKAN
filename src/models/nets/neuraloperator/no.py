# """
# @author   Maksim Penkin
# """

from torch import nn
from .hartley import HartleyLayer


class NeuralOperator2D(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, hid_ch=64, version="hartley", **kwargs):
        super(NeuralOperator2D, self).__init__()

        self.emb = nn.Linear(in_ch, hid_ch)

        if version == "hartley":
            # Stack of Hartley layers
            self.backbone = nn.Sequential(
                HartleyLayer(hid_ch, hid_ch, **kwargs),
                HartleyLayer(hid_ch, hid_ch, **kwargs),
                HartleyLayer(hid_ch, hid_ch, **kwargs)
            )
        else:
            raise NotImplementedError(f"Unrecognized `version` found: {version}.")

        self.restore = nn.Linear(hid_ch, out_ch)

    def forward(self, x):
        # x shape: (batch, spatial, in_channels)
        x = self.fc_in(x)  # (batch, spatial, hidden_channels)
        x = x.permute(0, 2, 1)  # (batch, hidden_channels, spatial)

        # Apply Hartley layers
        x = self.hartley_layers(x)

        x = x.permute(0, 2, 1)  # (batch, spatial, hidden_channels)
        x = self.fc_out(x)  # (batch, spatial, out_channels)
        return x
