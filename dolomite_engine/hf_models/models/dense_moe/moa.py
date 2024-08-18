from copy import deepcopy

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from ...enums import PositionEmbeddingType
from ...modeling_utils import SDPA, ParameterizedLinear, apply_rotary_pos_emb, repeat_key_value
from .config import DenseMoEConfig
from .inference import mask_probability


class DenseMoA_SDPA(SDPA):
    def __init__(
        self, config: DenseMoEConfig, causal: bool, layer_idx: int | None = None, inference_method: dict | None = None
    ) -> None:
        assert (
            config.n_head % config.num_experts == 0
        ), "number of attention heads must be divisible by the number of experts"
        self.heads_per_expert = config.n_head // config.num_experts

        self.num_experts = config.num_experts
        self.inference_method = inference_method

        config_copy = deepcopy(config)
        config_copy.num_key_value_heads = config_copy.num_experts
        super().__init__(config_copy, causal=causal, layer_idx=layer_idx)
        del config_copy

        self.gate = ParameterizedLinear(self.hidden_size, self.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

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
            scale=self.attention_multiplier if self.scale_attn_weights else 1,
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

        return attn_output
