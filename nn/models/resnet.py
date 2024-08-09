# """
# @author   Maksim Penkin
# """

from torchvision import models


def resnet18(weights="IMAGENET1K_V1", progress=True, **kwargs):
    return models.resnet18(weights=weights, progress=progress, **kwargs)


def resnet50(weights="IMAGENET1K_V2", progress=True, **kwargs):
    return models.resnet50(weights=weights, progress=progress, **kwargs)


def resnet101(weights="IMAGENET1K_V2", progress=True, **kwargs):
    return models.resnet101(weights=weights, progress=progress, **kwargs)
