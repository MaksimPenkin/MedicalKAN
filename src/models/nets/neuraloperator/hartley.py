# """
# @author   Dmitry Nesterov https://github.com/dmitrylala/denoising-fno/tree/master
# """

import numpy as np
import torch
from torch import nn

from ..layers import activate


def dht2d(x: torch.Tensor, is_inverse: bool = False) -> torch.Tensor:
    if not is_inverse:
        x_ft = torch.fft.fftshift(torch.fft.fft2(x, norm='backward'), dim=(-2, -1))
    else:
        x_ft = torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='backward')

    x_ht = x_ft.real - x_ft.imag

    if is_inverse:
        n = x.size()[-2:].numel()
        x_ht = x_ht / n

    return x_ht


def flip_periodic(x: torch.Tensor, axes: int | tuple | None = None) -> torch.Tensor:
    if axes is None:
        axes = (-2, -1)

    if isinstance(axes, int):
        axes = (axes,)

    return torch.roll(torch.flip(x, axes), shifts=(1,) * len(axes), dims=axes)


def contract_dense(x: torch.Tensor, w: nn.Parameter) -> torch.Tensor:
    # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
    return torch.einsum("bixy,ioxy->boxy", x, w)


class HartleySpectralConv2d(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int | tuple[int],
        factorization: str = 'dense',
        bias: bool = True,
        dtype: torch.dtype = torch.float,
        activation="relu",
        **_,  # noqa: ANN003
    ) -> None:
        super().__init__()

        if factorization != 'dense':
            msg = 'Supported only dense weight tensors'
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes  # n_modes is the total number of modes kept along each dimension

        # Create spectral weight tensor
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        self.weight = nn.Parameter(torch.rand(in_channels, out_channels, *tuple(np.array(self.n_modes) * 2), dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

        # Contraction function
        self._contract = contract_dense

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*((out_channels,) + (1,) * len(self.n_modes))),
            )

        self.activation = activate(activation)

    def hartley_conv(
        self,
        x: torch.Tensor,
        x_reverse: torch.Tensor,
        kernel: torch.Tensor,
        kernel_reverse: torch.Tensor,
    ) -> torch.Tensor:
        x_even = (x + x_reverse) / 2
        x_odd = (x - x_reverse) / 2
        return self._contract(x_even, kernel) + self._contract(x_odd, kernel_reverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        modes_h, modes_w = self.n_modes
        if h < 2 * modes_h or w < 2 * modes_w:
            msg = f'Expected input with bigger spatial dims: h>={w * modes_h}, w>={2 * modes_w}, got: {h=}, {w=}'  # noqa: E501
            raise ValueError(msg)

        x = dht2d(x)
        x_reverse = flip_periodic(x)

        center = tuple(s // 2 for s in x.size()[-2:])
        slices_x = [
            slice(None),
            slice(None),
            slice(center[0] - modes_h, center[0] + modes_h),
            slice(center[1] - modes_w, center[1] + modes_w),
        ]
        kernel = self.weight
        kernel_reverse = flip_periodic(kernel)
        total = self.hartley_conv(
            x[slices_x],
            x_reverse[slices_x],
            kernel,
            kernel_reverse,
        )

        # pad with zeros before idht
        pad = [
            (w - 2 * modes_w) // 2,
            (w - 2 * modes_w) // 2 + int(w % 2 == 1),
            (h - 2 * modes_h) // 2,
            (h - 2 * modes_h) // 2 + int(h % 2 == 1),
        ]
        x = torch.nn.functional.pad(total, pad, mode='constant', value=0)

        x = dht2d(x, is_inverse=True)

        if self.bias is not None:
            x = x + self.bias

        return self.activation(x)
