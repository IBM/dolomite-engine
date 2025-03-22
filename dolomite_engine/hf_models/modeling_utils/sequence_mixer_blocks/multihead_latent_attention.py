import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import divide_if_divisible, is_flash_attention_available
from ...enums import PositionEmbeddingType
from ..linear import ParameterizedLinear


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        query_compression_size: int,
        key_value_compression_size: int,
        num_attention_heads: int,
        attention_multiplier: float,
        position_embedding_type: PositionEmbeddingType,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.query_compression_size = query_compression_size
        self.key_value_compression_size = key_value_compression_size

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            raise NotImplementedError()
        else:
            self.c_attn_down_projection = ParameterizedLinear(
                self.hidden_size,
                self.query_compression_size + 2 * self.key_value_compression_size,
                bias=self.add_bias,
                std=std,
            )

            self.query_up_projection = ParameterizedLinear(
                self.query_compression_size, self.hidden_size, bias=self.add_bias, std=std
            )

            self.key_up_projection = ParameterizedLinear(
                self.key_value_compression_size, self.hidden_size, bias=self.add_bias, std=std
            )

            self.value_up_projection = ParameterizedLinear(
                self.key_value_compression_size, self.hidden_size, bias=self.add_bias, std=std
            )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert is_kernel_allowed(Kernel.flash_attention_2)
            assert past_key_values is None

        hidden_states = self.c_attn_down_projection(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            raise NotImplementedError()
        else:
            query, key, value = hidden_states.split(
                (self.query_compression_size, self.key_value_compression_size, self.key_value_compression_size), dim=-1
            )
            del hidden_states

            if past_key_values is not None:
                key, value = past_key_values.update(key, value, self.layer_idx)

            query = self.query_up_projection(query)
            key = self.key_up_projection(key)
            value = self.value_up_projection(value)

        if is_kernel_allowed(Kernel.flash_attention_2):
            if self.use_padding_free_transformer:
                total_q = query.shape[0]

                query = query.view(total_q, self.num_heads, -1)
                key = key.view(total_q, self.num_heads, -1)
                value = value.view(total_q, self.num_heads, -1)

                output_shape = (-1, self.hidden_size)
            else:
                batch_size, query_length = query.shape[:-1]
                key_length = key.shape[1]

                query = query.view(batch_size, query_length, self.num_heads, -1)
                key = key.view(batch_size, key_length, self.num_heads, -1)
                value = value.view(batch_size, key_length, self.num_heads, -1)

                output_shape = (batch_size, query_length, -1)

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            if self.use_padding_free_transformer:
                hidden_states = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self._get_softmax_scale(),
                    causal=self.causal,
                )
            else:
                hidden_states = _flash_attention_forward(
                    query_states=query,
                    key_states=key,
                    value_states=value,
                    attention_mask=attention_mask,
                    query_length=query_length,
                    is_causal=self.causal,
                    dropout=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self._get_softmax_scale(),
                )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(*output_shape)
        else:
            batch_size, query_length = query.shape[:-1]
            key_length = key.shape[1]

            query = query.view(batch_size, query_length, self.num_heads, -1).transpose(1, 2)
            key = key.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)
            value = value.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.softmax_dropout_p if self.training else 0,
                is_causal=self.causal if attention_mask is None else False,
                scale=self._get_softmax_scale(),
            )

            del query, key, value

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _get_softmax_scale(self) -> float:
        if self.attention_multiplier is None:
            softmax_scale = None
        else:
            softmax_scale = self.attention_multiplier

        return softmax_scale
