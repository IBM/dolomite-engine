from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import SDPA, apply_rotary_pos_emb, repeat_key_value
from .inference import mask_probability


class DenseMoA_SDPA(SDPA):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_experts: int,
        position_embedding_type: PositionEmbeddingType,
        causal: bool,
        add_bias: bool,
        scale_attention_weights: bool,
        attention_softmax_in_fp32: bool,
        scale_attention_softmax_in_fp32: bool,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_idx: int = None,
        inference_method: dict = None,
    ) -> None:
        assert (
            num_attention_heads % num_experts == 0
        ), "number of attention heads must be divisible by the number of experts"
        self.heads_per_expert = num_attention_heads // num_experts

        self.num_experts = num_experts
        self.inference_method = inference_method

        super().__init__(
            hidden_size,
            num_attention_heads,
            self.heads_per_expert,
            AttentionHeadType.gqa,
            position_embedding_type,
            causal,
            add_bias,
            scale_attention_weights,
            attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32,
            attn_pdrop,
            resid_pdrop,
            layer_idx,
        )

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # alibi_bias is already added in base model to the attention_mask
        assert not output_attentions

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=2)
            value = torch.cat((layer_past[1], value), dim=2)

        present = (key, value) if use_cache else None

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_heads, key_length, head_dim)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0,
            is_causal=self.causal if attention_mask is None else False,
            scale=None if self.scale_attn_weights else 1,
        )

        # ==========================================================================================
        # attn_output -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        attn_output = attn_output.transpose(1, 2)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads, head_dim)
        # ==========================================================================================

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        routing_weights = mask_probability(routing_weights, self.inference_method)

        routing_weights = routing_weights.unsqueeze(-2).unsqueeze(-1)

        # ==========================================================================================
        # routing_weights -> (batch_size, query_length, 1, num_experts, 1)
        # ==========================================================================================

        batch_size, query_length, _, _ = attn_output.shape
        attn_output = attn_output.view(batch_size, query_length, self.num_key_value_heads, self.num_experts, -1)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_key_value_heads, num_experts, head_dim)
        # ==========================================================================================

        attn_output = attn_output * routing_weights
        attn_output = attn_output.reshape(batch_size, query_length, -1)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present
