import torch.nn as nn

from ....utils import ProcessGroupManager
from ...modeling_utils_TP import get_attention_module_TP, get_module_placements, get_normalization_function_TP
from ..ladder_residual import LadderResidualConfig
from ..ladder_residual.layer import LadderResidualBlock


class LadderResidualBlock_TP(LadderResidualBlock):
    def __init__(
        self,
        config: LadderResidualConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
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
        self.mlp = LadderMLP_TP(
            config, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)
