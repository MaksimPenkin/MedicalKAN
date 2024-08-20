# """
# @author   Maksim Penkin
# """

from torch import nn
from torchvision import models


def conv1x1(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 1, **kwargs)


def conv3x3(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, **kwargs)


def conv5x5(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 5, padding=2, **kwargs)


def conv7x7(in_ch, out_ch, **kwargs):
    return nn.Conv2d(in_ch, out_ch, 7, padding=3, **kwargs)


class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = conv3x3(dim, dim)

        self.bn2 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))

        return identity + x


def resnet18(weights="IMAGENET1K_V1", progress=True, **kwargs):
    return models.resnet18(weights=weights, progress=progress, **kwargs)


def resnet50(weights="IMAGENET1K_V2", progress=True, **kwargs):
    return models.resnet50(weights=weights, progress=progress, **kwargs)


def resnet101(weights="IMAGENET1K_V2", progress=True, **kwargs):
    return models.resnet101(weights=weights, progress=progress, **kwargs)
