import torch
import torch.nn as nn
from transformers import DynamicCache

from ....utils import ProcessGroupManager
from ...modeling_utils_TP import get_normalization_function_TP
from ..desync_residual import DesyncResidualConfig
from .attention import get_attention_module
from .mlp import EnsembleMLP_TP
from .normalization import get_ensemble_normalization_function_TP


class DesyncResidualBlock_TP(nn.Module):
    def __init__(
        self,
        config: DesyncResidualConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        assert not sequence_parallel

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_rank = ProcessGroupManager.get_tensor_parallel_rank()

        self.previous_mlp_all_reduce = layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]
        self.current_attention_all_reduce = config.reduce_pattern[layer_idx]["attention"]

        if self.previous_mlp_all_reduce:
            self.ln_1 = get_normalization_function_TP(
                config.normalization_function,
                hidden_size,
                eps=config.layer_norm_epsilon,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        else:
            self.ln_1 = get_ensemble_normalization_function_TP(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )

        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )

        if self.current_attention_all_reduce:
            self.ln_2 = get_normalization_function_TP(
                config.normalization_function,
                hidden_size,
                eps=config.layer_norm_epsilon,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        else:
            self.ln_2 = get_ensemble_normalization_function_TP(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = EnsembleMLP_TP(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            residual,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states
