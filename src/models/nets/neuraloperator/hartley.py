# """
# @author   Maksim Penkin
# """

import torch
import torch.nn as nn
import torch.nn.functional as F


def hartley_transform(x):
    """
    Computes the Hartley Transform (real-valued alternative to Fourier).
    Args:
        x: Input tensor of shape (..., N), where N is the signal length.
    Returns:
        Hartley transform of x, same shape.
    """
    # FFT gives (a + jb), Hartley is a - b
    fft = torch.fft.fft(x, dim=-1)
    real, imag = fft.real, fft.imag
    return real - imag


def inverse_hartley_transform(x):
    """
    Inverse Hartley Transform (same as forward transform up to normalization).
    """
    return hartley_transform(x) / x.shape[-1]


class HartleyLayer(nn.Module):
    """
    Hartley Neural Operator layer (similar to Fourier Layer in FNO but real-valued).
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Hartley modes to keep

        # Learnable weights for Hartley space (real-valued)
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, modes, dtype=torch.float32))

    def forward(self, x):
        # x shape: (batch, channels, spatial)
        batch_size = x.shape[0]

        # Compute Hartley transform
        x_ht = hartley_transform(x)

        # Truncate high modes (keep low-frequency components)
        x_ht_trunc = x_ht[..., :self.modes]

        # Multiply by learnable weights in Hartley space
        out_ht_trunc = torch.einsum("bix,iox->box", x_ht_trunc, self.weights)

        # Pad zeros for discarded modes
        out_ht = F.pad(out_ht_trunc, (0, x.shape[-1] - self.modes))

        # Inverse Hartley transform to get spatial output
        x_out = inverse_hartley_transform(out_ht)

        return x_out
