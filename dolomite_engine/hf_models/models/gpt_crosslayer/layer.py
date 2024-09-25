import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import apply_rotary_pos_emb, get_normalization_function, repeat_key_value
from ..gpt_dolomite.mlp import MLP
from .attention import get_attention_module, get_key_value_projection
from .config import GPTCrossLayerConfig


class CrossLayer(nn.Module):
    def __init__(
        self,
        config: GPTCrossLayerConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.m_residual = config.m_residual

        self._use_eager_attention = attention_implementation == "eager"
        self._use_sdpa = attention_implementation == "sdpa"
        self._use_flash_attention_2 = attention_implementation == "flash_attention_2"
        self._use_padding_free_transformer = use_padding_free_transformer

        if self._use_padding_free_transformer:
            assert self._use_flash_attention_2, "padding free transformer only works with flash attention"

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        joint_residual: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(
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
        if joint_residual is not None:
            hidden_states = hidden_states + joint_residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        # residual connection
        hidden_states = hidden_states + residual
        return hidden_states


class GPTCrossLayerBlock(nn.Module):
    def __init__(
        self,
        config: GPTCrossLayerConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_indices: list[int],
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.num_heads
        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.joint_residual_stream = config.joint_residual_stream
        self.layer_idx = layer_idx

        self._use_eager_attention = attention_implementation == "eager"
        self._use_sdpa = attention_implementation == "sdpa"
        self._use_flash_attention_2 = attention_implementation == "flash_attention_2"
        self._use_padding_free_transformer = use_padding_free_transformer

        if self._use_padding_free_transformer:
            assert self._use_flash_attention_2, "padding free transformer only works with flash attention"

        self.kv_proj = get_key_value_projection(config, attention_implementation, use_padding_free_transformer)

        self.layers = nn.ModuleList(
            [
                CrossLayer(
                    config,
                    normalization_implementation,
                    attention_implementation,
                    use_padding_free_transformer,
                    layer_idx,
                )
                for layer_idx in layer_indices
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        joint_residual = hidden_states if self.joint_residual_stream else None
        key, value = self.kv_proj(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, layer_idx=self.layer_idx)

        if self._use_sdpa or self._use_eager_attention:
            key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
            value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

            if self._use_eager_attention:
                key = key.transpose(-1, -2)
                batch_size, _, _, key_length = key.shape
                key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)
        elif self._use_flash_attention_2:
            if not self._use_padding_free_transformer:
                if self.attention_head_type == AttentionHeadType.mqa:
                    key = key.squeeze(1).unsqueeze(2)
                    value = value.squeeze(1).unsqueeze(2)
                else:
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                key,
                value,
                joint_residual,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return hidden_states
