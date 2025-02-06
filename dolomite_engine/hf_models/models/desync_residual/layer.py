import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from .attention import get_sequence_mixer
from .config import DesyncResidualConfig
from .mlp import DesyncResidualMLP
from .normalization import get_desync_residual_normalization_function


class DesyncResidualBlock(nn.Module):
    def __init__(
        self,
        config: DesyncResidualConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.m_residual = config.m_residual

        self.previous_mlp_all_reduce = layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]
        self.current_attention_all_reduce = config.reduce_pattern[layer_idx]["attention"]

        if self.previous_mlp_all_reduce:
            self.ln_1 = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
        else:
            self.ln_1 = get_desync_residual_normalization_function(
                config.normalization_function,
                hidden_size,
                tp_world_size=config.pretraining_tensor_parallel_size,
                eps=config.layer_norm_epsilon,
            )

        self.sequence_mixer = get_sequence_mixer(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )

        if self.current_attention_all_reduce:
            self.ln_2 = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
        else:
            self.ln_2 = get_desync_residual_normalization_function(
                config.normalization_function,
                hidden_size,
                tp_world_size=config.pretraining_tensor_parallel_size,
                eps=config.layer_norm_epsilon,
            )

        self.mlp_block = DesyncResidualMLP(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self._prepare_hidden_states(hidden_states, self.previous_mlp_all_reduce)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.sequence_mixer(
            hidden_states,
            residual,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = self._prepare_hidden_states(hidden_states, self.current_attention_all_reduce)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states, residual)

        return hidden_states

    def _prepare_hidden_states(self, hidden_states: torch.Tensor, was_all_reduce_called: bool) -> torch.Tensor:
        if was_all_reduce_called:
            assert hidden_states.dim() == 3
            hidden_states = hidden_states.unsqueeze(0)
        else:
            assert hidden_states.dim() == 4

        return hidden_states
