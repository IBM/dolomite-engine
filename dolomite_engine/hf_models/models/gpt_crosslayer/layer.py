import torch
import torch.nn as nn
from transformers import DynamicCache

from ....utils import divide_if_divisible
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import apply_rotary_pos_emb, get_mlp_block, get_normalization_function, repeat_key_value
from .config import GPTCrossLayerConfig
from .sequence_mixers import get_key_value_projection, get_sequence_mixer


class GPTCrossLayerBlock(nn.Module):
    def __init__(
        self,
        config: GPTCrossLayerConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.layer_idx = layer_idx
        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self.attention_head_type = config.sequence_mixer_blocks[layer_idx].attention_head_type
        self.num_heads = config.num_attention_heads
        self.head_dim = divide_if_divisible(hidden_size, self.num_heads, "")
        self.num_key_value_heads = config.sequence_mixer_blocks[layer_idx].num_key_value_heads

        self._use_sdpa = attention_implementation == "sdpa"
        self._use_flash_attention_2 = attention_implementation == "flash_attention_2"
        self._use_padding_free_transformer = use_padding_free_transformer

        self.kv_proj = None
        if config.sharing_pattern[layer_idx] == layer_idx:
            self.kv_proj = get_key_value_projection(
                config,
                attention_implementation=attention_implementation,
                use_padding_free_transformer=use_padding_free_transformer,
                layer_idx=layer_idx,
            )

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.sequence_mixer = get_sequence_mixer(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp_block = get_mlp_block(
            config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.kv_proj is not None:
            key, value = self.kv_proj(hidden_states)

            if self.position_embedding_type == PositionEmbeddingType.rope:
                key = apply_rotary_pos_emb(key, rope_cos_sin)

            if past_key_values is not None:
                key, value = past_key_values.update(key, value, layer_idx=self.layer_idx)

            if self._use_sdpa:
                key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
                value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)
            elif self._use_flash_attention_2:
                if not self._use_padding_free_transformer:
                    if self.attention_head_type == AttentionHeadType.mqa:
                        key = key.squeeze(1).unsqueeze(2)
                        value = value.squeeze(1).unsqueeze(2)
                    else:
                        key = key.transpose(1, 2)
                        value = value.transpose(1, 2)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.sequence_mixer(
            hidden_states,
            key=key,
            value=value,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states, key, value
