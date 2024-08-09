# """
# @author   Maksim Penkin
# """

import torch
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)  # output.shape: [B, num_classes]; target.shape: [B].

        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)  # A namedtuple of (values, indices) is returned. indices.shape: [B, maxk].
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Expand (explicit broadcast) target downward to fit pred shape.

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mae(output, target):
    with torch.no_grad():
        return F.l1_loss(output, target)


def mse(output, target):
    with torch.no_grad():
        return F.mse_loss(output, target)
