from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.modules.activations import swish
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

from .....utils import is_einops_available, is_fla_available
from ....config import CommonConfig
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_normalization_function
from .utils import ParameterizedShortConvolution


if TYPE_CHECKING:
    from fla.models.utils import Cache


class GatedLinearAttention2(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        config: CommonConfig,
        mode: str = 'chunk',
        chunk_size: int = 64,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.chunk_size = chunk_size

        self.use_short_conv = config.use_short_conv
        self.use_output_gate = use_output_gate

        assert self.hidden_size % self.num_heads == 0, f"Hidden size {self.hidden_size} is not divisible by the number of heads {self.num_heads}."

        self.num_slot = int(self.hidden_size // self.num_heads)
        self.value_dim = self.hidden_size
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."

        initializer_range = config.initializer_range
        std_in = initializer_range
        init_method = InitMethod(config.init_method)
        if init_method == InitMethod.mup:
            std_in /= math.sqrt(config.m_width)
        self.q_proj = ParameterizedLinear(self.hidden_size, self.num_slot, bias=False, std=std_in)
        self.k_proj = ParameterizedLinear(self.hidden_size, self.num_slot, bias=False, std=std_in)

        if self.use_short_conv:
            std_conv = std_in
            self.conv_size = conv_size
            self.q_conv1d = ParameterizedShortConvolution(self.num_slot, conv_size, std=std_conv, activation=None)
            self.k_conv1d = ParameterizedShortConvolution(self.num_slot, conv_size, std=std_conv, activation=None)
            self.v_conv1d = ParameterizedShortConvolution(self.num_slot, conv_size, std=std_conv, activation=None)

        if self.use_output_gate:
            self.og_proj = ParameterizedLinear(self.value_dim, self.value_dim, bias=False, std=std_in)
            self.ig_proj = ParameterizedLinear(self.hidden_size, self.value_dim, bias=False, std=std_in)

        std_out = initializer_range / math.sqrt(2 * config.n_layer)
        if init_method == InitMethod.mup:
            std_out /= math.sqrt(config.m_width)
        self.o_proj = ParameterizedLinear(self.value_dim, self.hidden_size, bias=False, std=std_out)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.value_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.value_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state_q = last_state[0] if use_cache else None
            conv_state_k = last_state[1] if use_cache else None
            conv_state_vg = last_state[2] if use_cache else None
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = hidden_states
            q = self.q_conv1d(q, attention_mask, conv_state_q)
            k = self.k_conv1d(k, attention_mask, conv_state_k)
            v = self.v_conv1d(v, attention_mask, conv_state_vg)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = hidden_states

        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
            
        # improve precision
        k = k.float()

        q = F.softmax(q, dim=-1)
        k = F.sigmoid(k).to(v)
        gf = F.logsigmoid(-k)

        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gf, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, gf, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, gf, initial_state=recurrent_state, output_final_state=use_cache, scale=1)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                last_state = (conv_state_q, conv_state_k, conv_state_vg, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h l d -> b l (h d)')
        if self.use_output_gate:
            go = self.og_proj(o) + self.ig_proj(hidden_states)
            if self.fuse_norm_and_gate:
                o = self.g_norm_swish_gate(o, go)
            else:
                o = self.g_norm(o)
                o = o * self.gate_fn(go)
        else:
            o = self.g_norm(o)
        o = self.o_proj(o)

        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            state += (param.new_zeros(batch_size, self.num_slot, self.conv_size),
                        param.new_zeros(batch_size, self.num_slot, self.conv_size),
                        param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.num_slot * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size