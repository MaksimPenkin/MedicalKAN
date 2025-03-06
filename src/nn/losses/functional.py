# """
# @author   Maksim Penkin
# """

import torch.nn.functional as F


def accuracy(y_pred, y, topk=(1,)):
    batch_size = y.size(0)  # y_pred.shape: [B, num_classes]; y.shape: [B].

    _, pred = y_pred.topk(max(topk), dim=1, largest=True, sorted=True)  # A namedtuple of (values, indices) is returned. indices.shape: [B, maxk].
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))  # Expand (explicit broadcast) y downward to fit y_pred shape.

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mae(y_pred, y):
    return F.l1_loss(y_pred, y)


def mse(y_pred, y):
    return F.mse_loss(y_pred, y)
