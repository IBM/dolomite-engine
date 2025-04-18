import torch
import torch.nn as nn
from transformers import DynamicCache

from ..mixins import BaseBlock
from ..modeling_utils import ParameterizedLinear, get_mlp_block, get_normalization_function, get_sequence_mixer
from ..modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from .gpt_dolomite import GPTDolomiteConfig


class MTPBlock(BaseBlock):
    def __init__(
        self, config: GPTDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        nn.Module.__init__(self)

        self.is_mtp_block = True

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.mtp_blocks[layer_idx].sequence_mixer.sequence_mixer_type

        self.ln_3 = get_normalization_function(
            config.mtp_blocks[layer_idx].normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.sequence_mixer = get_sequence_mixer(
            config, True, use_padding_free_transformer, layer_idx=layer_idx, is_mtp_block=self.is_mtp_block
        )
        self.ln_4 = get_normalization_function(
            config.mtp_blocks[layer_idx].normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp_block = get_mlp_block(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            is_mtp_block=self.is_mtp_block,
            layer_idx=layer_idx,
        )

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )

        std = _get_std_for_linear(
            init_method=config.init_method, initializer_range=config.initializer_range, m_width=config.m_width
        )

        self.down_proj = ParameterizedLinear(
            2 * hidden_size,
            hidden_size,
            bias=config.mtp_blocks[layer_idx].add_bias,
            std=std,
        )

    def prepare_for_trm_block(
        self,
        x_emb: torch.Tensor,
        past_hidden_layer: torch.Tensor,
    ) -> torch.Tensor:

        x_emb_norm = self.ln_3(x_emb)
        past_hidden_norm = self.ln_4(past_hidden_layer)

        # concatenate
        fused_ip = torch.cat([past_hidden_norm, x_emb_norm], dim=-1)  # (B,L,2D)
        # Down proj
        hidden_states = self.down_proj(fused_ip)  # (B,L,D)
        return hidden_states

    def forward(
        self,
        x_emb: torch.Tensor,
        past_hidden_layer: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_for_trm_block(x_emb, past_hidden_layer)
        hidden_states = super().forward(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return hidden_states
