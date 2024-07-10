import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import CommonConfig
from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...utils import divide_if_divisible
from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.position_embedding import apply_rotary_pos_emb
from ...modeling_utils.attention.utils import repeat_key_value

# from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from einops import rearrange

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.activations import swish
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

from fla.models.utils import Cache


class GLA2Attention(nn.Module):

    def __init__(
        self,
        config: CommonConfig,
        layer_idx: int = None,
        mode: str = 'fused_chunk',
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head

        expand_ratio = self.hidden_size // self.num_heads
        self.expand_ratio = expand_ratio

        self.forget_dim = int(self.num_heads * self.expand_ratio)
        self.input_dim = self.hidden_size
        self.layer_idx = layer_idx

        assert self.mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{self.mode}`."
        assert self.forget_dim % self.num_heads == 0, f"forget dim must be divisible by num_heads of {self.num_heads}"
        assert self.input_dim % self.num_heads == 0, f"input dim must be divisible by num_heads of {self.num_heads}"

        self.head_f_dim = self.expand_ratio
        self.head_i_dim = self.hidden_size // self.num_heads

        initializer_range = config.initializer_range
        std = initializer_range
        init_method = InitMethod(config.init_method)
        if init_method == InitMethod.mup:
            std /= math.sqrt(config.m_width)
        self.q_proj = ParameterizedLinear(
            self.hidden_size, self.forget_dim, 
            bias=False, std=std
            )
        self.f_proj = ParameterizedLinear(
            self.hidden_size, self.forget_dim, 
            bias=False, std=std
            )
        self.i_proj = ParameterizedLinear(
            self.hidden_size, self.input_dim, 
            bias=False, std=std
            )
        # self.g_norm = RMSNorm(self.hidden_size, elementwise_affine=True, eps=1e-5)

        std = initializer_range / math.sqrt(2 * config.n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(config.m_width)
        self.o_proj = ParameterizedLinear(
            self.input_dim, self.hidden_size, 
            bias=False, std=std
            )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
    ):
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        q = self.q_proj(hidden_states)
        f = self.f_proj(hidden_states)
        i = self.i_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))

        q = F.softmax(q, dim=-1)
        # the lower bound for the first layer is zero
        k, g = 1 - F.softmax(f, dim=-1), F.log_softmax(f, dim=-1)
        q, k, i, g = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, i, g))

        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.o_proj(o)

        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.forget_dim, self.conv_size),
                          param.new_zeros(batch_size, self.forget_dim, self.conv_size),
                          param.new_zeros(batch_size, self.input_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_f_dim, self.head_i_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.forget_dim * self.head_i_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
