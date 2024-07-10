# -*- coding: utf-8 -*-

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import CommonConfig

from typing import Optional, Tuple

from einops import rearrange

from fla.modules import RMSNorm
from fla.ops.delta_rule import (chunk_delta_rule, fused_chunk_delta_rule,
                                fused_recurrent_linear_attn_delta_rule)
from fla.models.utils import Cache


def simple_norm(x):
    return (F.normalize(x, dim=-1) * x.shape[-1] ** 0.5).to(x)


# @torch.jit.script
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


# @torch.jit.script
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


# @torch.jit.script
def elu_norm(x):
    dtype = x.dtype
    x = F.elu(x, 1., False) + 1.
    return (x / x.sum(-1, keepdim=True)).to(dtype)


# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1
class DeltaNet(nn.Module):
    def __init__(
        self,
        config: CommonConfig,
        layer_idx: int = None,
        mode: str = 'fused_chunk',
        chunk_size: int = 16,
        use_beta: bool = True,
        use_output_norm: bool = True,
        use_elu: bool = False,
        qk_activation: str = 'silu',
        qk_norm: str = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.chunk_size = chunk_size
        self.use_output_norm = use_output_norm

        self.key_dim = self.hidden_size
        self.value_dim = self.hidden_size 
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        self.use_elu = use_elu
        if self.use_beta:
            self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

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
        **kwargs
    ):
        # change to inference mode.
        mode = 'fused_recurrent' if hidden_states.shape[1] < 64 else self.mode

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            if attention_mask.shape[-1] != hidden_states.shape[-2]:
                attention_mask = attention_mask[:, -1:]

        q = (self.q_proj(hidden_states))
        k = (self.k_proj(hidden_states))
        v = self.silu(self.v_proj(hidden_states))

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm is not None:
            if self.qk_norm == 'l2':
                k = nn.functional.normalize(k, dim=-1, p=2).to(v)  # auto mixed precision type transfer is annoying.
                q = nn.functional.normalize(q, dim=-1, p=2).to(v)
            elif self.qk_norm == 'sum':
                q = sum_norm(q).to(v)
                k = sum_norm(k).to(v)

        if self.use_beta:
            beta = rearrange(self.b_proj(hidden_states), 'b l h -> b h l').sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])
        state = past_key_values[self.layer_idx][-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_linear_attn_delta_rule(q, k, v, beta, state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            assert self.chunk_size in [16, 32, 64]
            o, recurrent_state = fused_chunk_delta_rule(q, k, v, beta, self.chunk_size, state, output_final_state=use_cache)
        elif mode == 'chunk':
            assert self.chunk_size in [16, 32, 64]
            o, recurrent_state = chunk_delta_rule(q, k, v, beta, self.chunk_size, state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            state = (recurrent_state,)
            past_key_values.update(state, self.layer_idx)

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                # for q/k/v each
                state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state