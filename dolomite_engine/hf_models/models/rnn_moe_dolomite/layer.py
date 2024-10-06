from copy import deepcopy

import torch.nn as nn

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.mlp import MLP
from ..moe_dolomite.layer import SparseMoEBlock
from ..moe_dolomite.moe import get_moe
from ..rnn_dolomite.attention import DeltaNet, RNNFlashAttention2
from .config import RNNMoEDolomiteConfig


class RNNMoEDolomiteBlock(SparseMoEBlock):
    def __init__(
        self,
        config: RNNMoEDolomiteConfig,
        normalization_implementation: str,
        attention_pattern: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        assert not use_padding_free_transformer

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        if attention_pattern == "flash_attention_2":
            self.attn = RNNFlashAttention2(config, True, layer_idx)
        elif attention_pattern == "deltanet":
            self.attn = DeltaNet(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Attention pattern {attention_pattern} not supported.")

        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        self.moe = get_moe(
            config,
            moe_implementation=moe_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )

        self.mlp = None
        if config.shared_n_inner is not None:
            shared_config = deepcopy(config)
            shared_config.n_inner = config.shared_n_inner
            self.mlp = MLP(shared_config)
            del shared_config
