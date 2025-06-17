# """
# @author   Maksim Penkin
# """

from torch import nn


def conv1x1(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 1, **kwargs)


def conv3x3(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, **kwargs)


def conv5x5(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 5, padding=2, **kwargs)


def conv7x7(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 7, padding=3, **kwargs)


def fc(in_ch, out_ch, **kwargs):
    return nn.Linear(in_ch, out_ch, **kwargs)


def activate(activation=None):
    if not activation:
        return nn.Identity()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "Tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError(f"Unrecognized `activation` found: {activation}.")


class ResBlock(nn.Module):
    """Pre-activated residual block.

    Pre-activation ResNet is a variant of the original Residual Network (ResNet) architecture
    that modifies the order of operations within the residual blocks to improve training and performance.
    Introduced by Kaiming He et al., this architecture aims to address the issues related to gradient flow
    in deep networks by changing the placement of activation functions and Batch Normalization layers.

    Args:
        in_ch: Input feature dimension.
        out_ch: Output feature dimension. If not specified, in_ch is used.
        hid_ch: Hidden feature dimension. If not specified, min(in_ch, out_ch) is used.
        bn: If True, batch normalization is applied.
        layer: Layer to be used. Either nn.Module, or string (e.g. "conv3x3").
    """

    def __init__(self, in_ch, out_ch=None, hid_ch=None, bn=False, layer=None, **kwargs):
        super(ResBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        if hid_ch is None:
            hid_ch = min(in_ch, out_ch)

        layer = self._make_layer(layer)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
            nn.ReLU(),
            layer(in_ch, hid_ch, **kwargs)
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(hid_ch) if bn else nn.Identity(),
            nn.ReLU(),
            layer(hid_ch, out_ch, **kwargs)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
                layer(in_ch, out_ch, bias=False, **kwargs)
            )
        else:
            self.shortcut = nn.Identity()

    def _make_layer(self, layer):
        if layer is None:
            return conv3x3

        if callable(layer):
            return layer
        if layer == "fc":
            return fc
        elif layer == "conv1x1":
            return conv1x1
        elif layer == "conv3x3":
            return conv3x3
        elif layer == "conv5x5":
            return conv5x5
        elif layer == "conv7x7":
            return conv7x7
        else:
            raise ValueError(f"Unrecognized layer found: {layer}.")

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = self.block2(x)

        return self.shortcut(identity) + x
