import torch.nn as nn
from torch import Tensor
from transformers import DynamicCache

from ...enums import AttentionHeadType
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
        self.ln_2 = get_ensemble_normalization_function(
            config.normalization_function,
            hidden_size,
            tp_world_size=config.pretraining_tensor_parallel_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = EnsembleMLP(config, layer_idx=layer_idx)
