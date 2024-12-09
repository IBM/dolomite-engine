import torch.nn as nn

from ...modeling_utils_TP import get_attention_module_TP, get_normalization_function_TP
from ..moe_dolomite import MoEDolomiteConfig
from ..moe_dolomite.layer import MoEDolomiteBlock
from .moe_TP import ScatterMoE_TP


class MoEDolomiteBlock_TP(MoEDolomiteBlock):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
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
