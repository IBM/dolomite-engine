import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .attention import get_attention_module
from .config import GPTEnsembleConfig
from .mlp import EnsembleMLP
from .normalization import get_ensemble_normalization_function


class GPTEnsembleBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: GPTEnsembleConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual
        self.reduce_pattern = config.reduce_pattern

        if layer_idx == 0 or config.reduce_pattern[layer_idx - 1]["mlp"]:
            self.ln_1 = get_normalization_function(
                config.normalization_function,
                hidden_size,
                eps=config.layer_norm_epsilon,
                normalization_implementation=normalization_implementation,
            )
        else:
            self.ln_1 = get_ensemble_normalization_function(
                config.normalization_function,
                hidden_size,
                tp_world_size=config.pretraining_tensor_parallel_size,
                eps=config.layer_norm_epsilon,
                normalization_implementation=normalization_implementation,
            )

        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )

        if config.reduce_pattern[layer_idx]["attention"]:
            self.ln_2 = get_normalization_function(
                config.normalization_function,
                hidden_size,
                eps=config.layer_norm_epsilon,
                normalization_implementation=normalization_implementation,
            )
        else:
            self.ln_2 = get_ensemble_normalization_function(
                config.normalization_function,
                hidden_size,
                tp_world_size=config.pretraining_tensor_parallel_size,
                eps=config.layer_norm_epsilon,
                normalization_implementation=normalization_implementation,
            )

        self.mlp = EnsembleMLP(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if self.layer_idx == 0 or self.reduce_pattern[self.layer_idx - 1]["mlp"]:
            assert hidden_states.dim() == 3
            hidden_states = hidden_states.unsqueeze(0)
        else:
            assert hidden_states.dim() == 4

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

        if self.reduce_pattern[self.layer_idx]["attention"]:
            assert hidden_states.dim() == 3
            hidden_states = hidden_states.unsqueeze(0)
        else:
            assert hidden_states.dim() == 4

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states
