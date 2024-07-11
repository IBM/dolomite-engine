# -*- coding: utf-8 -*-

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.models.utils import Cache
from fla.modules import RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_linear_attn_delta_rule

from ...config import CommonConfig
from ...enums import InitMethod
from ...modeling_utils.linear import ParameterizedLinear


def simple_norm(x):
    return (F.normalize(x, dim=-1) * x.shape[-1] ** 0.5).to(x)


# @torch.jit.script
def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


# @torch.jit.script
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


# @torch.jit.script
def elu_norm(x):
    dtype = x.dtype
    x = F.elu(x, 1.0, False) + 1.0
    return (x / x.sum(-1, keepdim=True)).to(dtype)


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


# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1
class DeltaNet(nn.Module):
    def __init__(
        self,
        config: CommonConfig,
        layer_idx: int = None,
        mode: str = "chunk",
        chunk_size: int = 64,
        use_beta: bool = True,
        use_output_norm: bool = True,
        use_elu: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        use_short_conv: bool = True,
        conv_size: int = 4,
        share_conv_kernel: bool = False,
    ):
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_short_conv = use_short_conv
        self.share_conv_kernel = share_conv_kernel
        self.initializer_range = config.initializer_range

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

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

        assert mode in ["chunk", "fused_chunk", "fused_recurrent"], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"

        initializer_range = config.initializer_range
        std_in = initializer_range
        init_method = InitMethod(config.init_method)
        if init_method == InitMethod.mup:
            std_in /= math.sqrt(config.m_width)
        self.q_proj = ParameterizedLinear(self.hidden_size, self.key_dim, bias=False, std=std_in)
        self.k_proj = ParameterizedLinear(self.hidden_size, self.key_dim, bias=False, std=std_in)
        self.v_proj = ParameterizedLinear(self.hidden_size, self.value_dim, bias=False, std=std_in)

        if use_short_conv:
            std_conv = initializer_range
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ParameterizedShortConvolution(
                    self.hidden_size, conv_size, activation=None, std=std_conv
                )
            else:
                self.q_conv1d = ParameterizedShortConvolution(
                    self.key_dim, conv_size, activation="silu" if qk_activation == "silu" else None, std=std_conv
                )
                self.k_conv1d = ParameterizedShortConvolution(
                    self.key_dim, conv_size, activation="silu" if qk_activation == "silu" else None, std=std_conv
                )
                self.v_conv1d = ParameterizedShortConvolution(
                    self.value_dim, conv_size, activation="silu", std=std_conv
                )

        self.use_beta = use_beta
        self.use_elu = use_elu
        if self.use_beta:
            self.b_proj = ParameterizedLinear(self.hidden_size, self.num_heads, bias=False, std=std_in)
        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        std_out = initializer_range / math.sqrt(2 * config.n_layer)
        if init_method == InitMethod.mup:
            std_out /= math.sqrt(config.m_width)
        self.o_proj = ParameterizedLinear(self.value_dim, self.hidden_size, bias=False, std=std_out)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.initializer_range
        self.q_conv1d.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ):
        # change to inference mode.
        mode = "fused_recurrent" if hidden_states.shape[1] < 64 else self.mode
        use_cache = (past_key_values is not None) and (len(past_key_values) > self.layer_idx)

        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            if attention_mask.shape[-1] != hidden_states.shape[-2]:
                attention_mask = attention_mask[:, -1:]

        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)

                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_k = last_state[1] if use_cache else None
                conv_state_v = last_state[2] if use_cache else None

                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

                q = self.q_conv1d(q, attention_mask, conv_state_q)
                k = self.k_conv1d(k, attention_mask, conv_state_k)
                v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.silu(self.v_proj(hidden_states))

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        q, k, v = map(lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads), (q, k, v))

        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == "identity":
                pass
            else:
                raise NotImplementedError

        if self.qk_norm is not None:
            if self.qk_norm == "l2":
                k = nn.functional.normalize(k, dim=-1, p=2).to(v)  # auto mixed precision type transfer is annoying.
                q = nn.functional.normalize(q, dim=-1, p=2).to(v)
            elif self.qk_norm == "sum":
                q = sum_norm(q).to(v)
                k = sum_norm(k).to(v)

        if self.use_beta:
            beta = rearrange(self.b_proj(hidden_states), "b l h -> b h l").sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])
        state = past_key_values[self.layer_idx][-1] if use_cache else None
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_linear_attn_delta_rule(
                q, k, v, beta, state, output_final_state=use_cache
            )
        elif mode == "fused_chunk":
            assert self.chunk_size in [16, 32, 64]
            o, recurrent_state = fused_chunk_delta_rule(
                q, k, v, beta, self.chunk_size, state, output_final_state=use_cache
            )
        elif mode == "chunk":
            assert self.chunk_size in [16, 32, 64]
            o, recurrent_state = chunk_delta_rule(q, k, v, beta, self.chunk_size, state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    state = (conv_state, recurrent_state)
                else:
                    state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                state = (recurrent_state,)
            state = (recurrent_state,)
            past_key_values.update(state, self.layer_idx)

        o = rearrange(o, "b h l d -> b l h d")
        o = self.o_norm(o)
        o = rearrange(o, "b l h d -> b l (h d)")
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
                state += (
                    param.new_zeros(batch_size, self.key_dim, self.conv_size),
                    param.new_zeros(batch_size, self.key_dim, self.conv_size),
                    param.new_zeros(batch_size, self.value_dim, self.conv_size),
                )
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state
