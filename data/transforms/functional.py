# """
# @author   Maksim Penkin
# """

import torchvision.transforms.functional as F


def to_tensor(x):
    return F.to_tensor(x)
