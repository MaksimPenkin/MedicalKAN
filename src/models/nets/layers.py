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


def _make_layer(layer, *args, **kwargs):
    if layer == "conv1x1":
        return conv1x1(*args, **kwargs)
    elif layer == "conv3x3":
        return conv3x3(*args, **kwargs)
    elif layer == "conv5x5":
        return conv5x5(*args, **kwargs)
    elif layer == "conv7x7":
        return conv7x7(*args, **kwargs)
    elif layer == "fc":
        return fc(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognized `layer` found: {layer}.")


class ResBlock(nn.Module):
    """Pre-activated residual block.

    Pre-activation ResNet is a variant of the original Residual Network (ResNet) architecture
    that modifies the order of operations within the residual blocks to improve training and performance.
    Introduced by Kaiming He et al., this architecture aims to address the issues related to gradient flow
    in deep networks by changing the placement of activation functions and batch normalization layers.

    Args:
        in_ch: Input feature dimension.
        out_ch: Output feature dimension. If not specified, in_ch is used.
        hid_ch: Hidden feature dimension. If not specified, min(in_ch, out_ch) is used.
        bn: If True, batch normalization is applied.
        layer: Layer to be used (deserialized by internal def _make_layer(...)).
    """

    def __init__(self, in_ch, out_ch=None, hid_ch=None, bn=False, layer="conv3x3", **kwargs):
        super(ResBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch
        if hid_ch is None:
            hid_ch = min(in_ch, out_ch)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
            nn.ReLU(),
            _make_layer(layer, in_ch, hid_ch, **kwargs)
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(hid_ch) if bn else nn.Identity(),
            nn.ReLU(),
            _make_layer(layer, hid_ch, out_ch, **kwargs)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_ch) if bn else nn.Identity(),
                _make_layer(layer, in_ch, out_ch, bias=False, **kwargs)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = self.block2(x)

        return self.shortcut(identity) + x


class ConvBlock(nn.Module):
    """Pre-activated convolutional block.

    The block activates inputs, applies convolution and adds skip connection, if specified.

    Args:
        in_ch: Input feature dimension.
        out_ch: Output feature dimension. If not specified, in_ch is used.
        bn: If True, batch normalization is applied.
        layer: Layer to be used (deserialized by internal def _make_layer(...)).
        activation: Activation to be used (deserialized by internal def activate(...)).
    """

    def __init__(self, in_ch, out_ch=None, bn=False, layer="conv3x3", activation="relu", **kwargs):
        super(ConvBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch

        self.bn = nn.BatchNorm2d(in_ch) if bn else nn.Identity()
        self.act = activate(activation)
        self.conv = _make_layer(layer, in_ch, out_ch, **kwargs)

    def forward(self, x, skip=None, return_feature=False):
        feat = self.act(self.bn(x))
        x = self.conv(feat)
        if skip is not None:
            x += skip
        return x if not return_feature else (x, feat)


class ResidualEncoderBlock(nn.Module):
    """Residual encoder block.

    The block generates features by ResBlock and encodes (projects) features to the output feature dimension via conv3x3 with stride 2.

    Args:
        in_ch: Input feature dimension.
        out_ch: Output feature dimension. If not specified, in_ch is used.
    """

    def __init__(self, in_ch, out_ch=None, **kwargs):
        super(ResidualEncoderBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch

        self.feat = ResBlock(in_ch, **kwargs)
        self.down = conv3x3(in_ch, out_ch, stride=2)

    def forward(self, x, return_feature=False):
        feat = self.feat(x)
        x = self.down(feat)
        return x if not return_feature else (x, feat)


class ResidualDecoderBlock(nn.Module):
    """Residual decoder block.

    The block decodes (up-projects) features to the output feature dimension via bilinear upsample and conv3x3,
    applies skip connection, if specified, and generates features by ResBlock.

    Args:
        in_ch: Input feature dimension.
        out_ch: Output feature dimension. If not specified, in_ch is used.
    """

    def __init__(self, in_ch, out_ch=None, **kwargs):
        super(ResidualDecoderBlock, self).__init__()

        if out_ch is None:
            out_ch = in_ch

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv3x3(in_ch, out_ch)
        )
        self.feat = ResBlock(out_ch, **kwargs)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x += skip
        x = self.feat(x)
        return x
