# """
# @author   Maksim Penkin
# """

import torchvision.transforms.functional as F

from .base_transform import ITransform


class ToTensor(ITransform):

    def __init__(self):
        super(ToTensor, self).__init__()

    def _fn(self, x):
        return F.to_tensor(x)
