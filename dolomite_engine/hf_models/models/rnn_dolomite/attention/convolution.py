import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....utils import is_causal_conv1d_available, is_einops_available


if is_einops_available():
    from einops import rearrange


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


class ParameterizedShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: nn.Module = nn.Identity(),
        use_fast_conv1d: bool = True,
        std: float | None = None,
    ) -> None:
        self.std = std

        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

        self.hidden_size = hidden_size
        self.activation = activation

        if not is_causal_conv1d_available():
            if use_fast_conv1d:
                raise RuntimeError(
                    "Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel "
                    "or set `use_fast_conv1d` to False"
                )
            else:
                warnings.warn(
                    "The naive Pytorch verison is very slow in practice, "
                    "please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel",
                    category=ImportWarning,
                )
        self.use_fast_conv1d = use_fast_conv1d

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        if not self.use_fast_conv1d:
            s += ", use_fast_conv1d={use_fast_conv1d}"
        return s.format(**self.__dict__)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[batch_size, seq_len, hidden_size]`
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[batch_size, hidden_size, kernel_size]`,
        Returns:
            Tensor of shape `[batch_size, seq_len, hidden_size]`. The `cache` (if provided) is updated inplace.
        """

        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)
        x = rearrange(x, "b l d -> b d l")
        # Update state (B D W)
        if cache is not None:
            cache.copy_(F.pad(x, (self.kernel_size[0] - x.shape[-1], 0)))
        if self.use_fast_conv1d:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., : x.shape[-1]]
            x = self.activation(x)
        return rearrange(x, "b d l -> b l d")

    def step(self, x: torch.Tensor, cache: torch.Tensor):
        assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            x = self.activation(x).to(dtype)
        return x.unsqueeze(1)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size

    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if self.bias is not None:
                self.bias.zero_()
