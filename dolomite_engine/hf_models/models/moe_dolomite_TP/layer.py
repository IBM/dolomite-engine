from copy import deepcopy

import torch.nn as nn

from ...enums import AttentionHeadType
from ...modeling_utils_TP import get_attention_module_TP, get_normalization_function_TP
from ..gpt_dolomite_TP.layer import MLP_TP
from ..moe_dolomite import MoEDolomiteConfig
from ..moe_dolomite.layer import SparseMoEBlock
from .moe_TP import ScatterMoE_TP


class SparseMoEBlock_TP(SparseMoEBlock):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
        self.attn = get_attention_module_TP(
            config,
            True,
            attention_implementation=attention_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
        )
        self.ln_2 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        assert moe_implementation == "scattermoe", "TP for MoE is only implemented with scattermoe"
        self.moe = ScatterMoE_TP(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )

        self.mlp = None
        if config.shared_n_inner is not None:
            shared_config = deepcopy(config)
            shared_config.n_inner = config.shared_n_inner
            self.mlp = MLP_TP(shared_config)
            del shared_config
