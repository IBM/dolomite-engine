from typing import Optional

import torch.nn as nn

from .....utils import is_einops_available, is_fla_available


if is_einops_available():
    from einops import rearrange

if is_fla_available():
    from fla.modules import ShortConvolution
    from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_linear_attn_delta_rule


class ParameterizedShortConvolution(ShortConvolution):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = "silu",
        use_causal_conv: Optional[bool] = True,
        std: Optional[float] = None,
    ):
        self.std = std
        super().__init__(hidden_size, kernel_size, bias, activation, use_causal_conv)

    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if self.bias is not None:
                self.bias.zero_()
