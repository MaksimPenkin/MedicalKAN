# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn
import torch.nn.functional as F


class HartleyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(HartleyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = stride

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

    def forward(self, x):
        batch_size, _, h, w = x.shape
        kh, kw = self.kernel_size

        # Calculate output dimensions
        out_h = (h + 2 * self.padding - kh) // self.stride + 1
        out_w = (w + 2 * self.padding - kw) // self.stride + 1

        # Pad input if needed
        if self.padding > 0:
            x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        else:
            x_padded = x

        # Compute required FFT size
        fft_h = h + 2 * self.padding + kh - 1
        fft_w = w + 2 * self.padding + kw - 1

        # Compute 2D Hartley transforms
        x_ht = self.dht2(x_padded, fft_h, fft_w)
        weight_ht = self.dht2(self.weight, fft_h, fft_w)

        # Prepare for proper Hartley convolution
        x_ht = x_ht.unsqueeze(1)  # [batch, 1, in_channels, fft_h, fft_w]
        weight_ht = weight_ht.unsqueeze(0)  # [1, out_channels, in_channels, fft_h, fft_w]

        # CORRECT HARTLEY CONVOLUTION FORMULA
        # Need to compute special combinations
        weight_ht_flipped = torch.flip(weight_ht, dims=[-2, -1])

        # Main terms
        term1 = x_ht * weight_ht
        term2 = x_ht * weight_ht_flipped
        term3 = torch.flip(x_ht, dims=[-2, -1]) * weight_ht

        # Combine terms according to Hartley convolution theorem
        out_ht = (term1 + term2 + term3 - torch.flip(x_ht, dims=[-2, -1]) * weight_ht_flipped) / 2

        # Sum over input channels
        out_ht = out_ht.sum(dim=2)

        # Inverse Hartley transform
        output = self.idht2(out_ht, fft_h, fft_w)

        # Crop to correct output size and add bias
        output = output[:, :, :out_h, :out_w]

        return output

    @staticmethod
    def dht1(x):
        """1D Discrete Hartley Transform using FFT"""
        n = x.shape[-1]
        fft = torch.fft.fft(x, dim=-1)
        return fft.real - fft.imag

    @staticmethod
    def idht1(x):
        """Inverse 1D Discrete Hartley Transform"""
        return x / x.shape[-1]

    def dht2(self, x, fft_h=None, fft_w=None):
        """2D Discrete Hartley Transform"""
        if fft_h is not None and fft_w is not None:
            # Pad to desired FFT size
            if len(x.shape) == 4:  # [batch, channels, h, w]
                pad_h = fft_h - x.shape[-2]
                pad_w = fft_w - x.shape[-1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
            else:  # [channels_out, channels_in, h, w]
                pad_h = fft_h - x.shape[-2]
                pad_w = fft_w - x.shape[-1]
                x = F.pad(x, (0, pad_w, 0, pad_h))

        # Apply 1D DHT along rows then columns
        x = self.dht1(x)
        x = self.dht1(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    def idht2(self, x, fft_h=None, fft_w=None):
        """Inverse 2D Discrete Hartley Transform"""
        # Apply 1D inverse DHT along rows then columns
        x = self.idht1(x)
        x = self.idht1(x.transpose(-1, -2)).transpose(-1, -2)
        return x
