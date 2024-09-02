import torch.nn as nn

from ....utils import ProcessGroupManager
from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from ..gpt_ensemble import GPTEnsembleConfig
from .attention import get_attention_module
from .mlp import EnsembleMLP_TP


class GPTEnsembleBlock_TP(GPTDolomiteBlock):
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
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_rank = ProcessGroupManager.get_tensor_parallel_rank()

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = EnsembleMLP_TP(config, layer_idx=layer_idx)
