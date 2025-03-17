import torch
import torch.nn as nn
from transformers import DynamicCache

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_cute_kernels_available
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from .config import GPTDolomiteConfig


if is_cute_kernels_available():
    from cute_kernels import fused_residual_add_rmsnorm_cute


class GPTDolomiteBlock(nn.Module):
    def __init__(
        self,
        config: GPTDolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type
        self.layer_idx = layer_idx
        self.num_layers = config.num_layers

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
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        use_fused_residual_add_rmsnorm = is_kernel_allowed(Kernel.fused_residual_add_rmsnorm_cute)

        if use_fused_residual_add_rmsnorm:
            if self.layer_idx == 0:
                residual = hidden_states
                hidden_states = self.ln_1(hidden_states)
            else:
                hidden_states, residual = fused_residual_add_rmsnorm_cute(
                    x=hidden_states,
                    residual=residual,
                    weight=self.ln_1.weight,
                    eps=self.ln_1.eps,
                    multiplier=self.m_residual,
                )
        else:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)

        hidden_states = self._sequence_mixer_forward(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if use_fused_residual_add_rmsnorm:
            hidden_states, residual = fused_residual_add_rmsnorm_cute(
                x=hidden_states,
                residual=residual,
                weight=self.ln_2.weight,
                eps=self.ln_2.eps,
                multiplier=self.m_residual,
            )
        else:
            if self.m_residual is not None:
                hidden_states = hidden_states * self.m_residual

            hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states)

        if (
            use_fused_residual_add_rmsnorm and self.layer_idx == self.num_layers - 1
        ) or not use_fused_residual_add_rmsnorm:
            if self.m_residual is not None:
                hidden_states = hidden_states * self.m_residual

            hidden_states = hidden_states + residual
            residual = None

        return hidden_states, residual

    def _sequence_mixer_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sequence_mixer_type in ["softmax_attention", "stickbreaking_attention"]:
            hidden_states = self.sequence_mixer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type == "mamba2":
            hidden_states = self.sequence_mixer(
                hidden_states, cache_params=past_key_values, attention_mask=attention_mask
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({self.sequence_mixer_type})")

        return hidden_states
