import torch
import torch.nn as nn
from transformers import DynamicCache

from ....utils import ProcessGroupManager
from ...enums import InitMethod
from ...modeling_utils_TP import get_normalization_function_TP
from ..desync_residual import DesyncResidualConfig
from .attention import get_sequence_mixer_TP
from .mlp import DesyncResidualMLP_TP
from .normalization import get_desync_residual_normalization_function_TP


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
            self.ln_1 = get_desync_residual_normalization_function_TP(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )

        self.sequence_mixer = get_sequence_mixer_TP(
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
            self.ln_2 = get_desync_residual_normalization_function_TP(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )

        block = config.mlp_blocks[layer_idx]

        self.mlp_block = DesyncResidualMLP_TP(
            hidden_size=hidden_size,
            intermediate_size=block.intermediate_size,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=InitMethod(config.init_method),
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            m_residual=config.m_residual,
            num_layers=config.num_layers,
            pretraining_tensor_parallel_size=config.pretraining_tensor_parallel_size,
            all_reduce=layer_idx == config.num_layers - 1 or config.reduce_pattern[layer_idx]["mlp"],
            attention_did_all_reduce=config.reduce_pattern[layer_idx]["attention"],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states, residual)

        return hidden_states
