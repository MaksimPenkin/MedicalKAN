# """
# @author   Maksim Penkin
# """

import torch
import torch.nn.functional as F


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    n, c, h, w = x.size()
    return F.pixel_shuffle(x, block_size)
