# """
# @author   Maksim Penkin
# """

import torchvision.transforms.functional as F

from data.transforms.base_transform import ITransform


class ToTensor(ITransform):

    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, *args):
        return tuple(F.to_tensor(x) for x in args)
