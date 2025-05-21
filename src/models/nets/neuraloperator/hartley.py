# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn


def ht2(x):
    fft = torch.fft.fft2(x)
    real, imag = fft.real, fft.imag
    return real - imag


def iht2(x):
    return 1 / (x.shape[-1] * x.shape[-2]) * x


class HartleyConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1=16, modes2=16):
        super(HartleyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float32))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ht = ht2(x)

        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1), device=x.device, dtype=torch.float32)
        out_ht[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ht[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ht[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ht[:, :, -self.modes1:, -self.modes2:] = self.compl_mul2d(x_ht[:, :, -self.modes1:, -self.modes2:], self.weights1)
        out_ht[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(x_ht[:, :, :self.modes1, -self.modes2:], self.weights2)

        x = iht2(out_ht)

        return x
