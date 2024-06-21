from ...enums import AttentionHeadType
from ...modeling_utils import get_attention_module, get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .config import GPTEnsembleConfig
from .mlp import EnsembleMLP


class GPTEnsembleBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: GPTEnsembleConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

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
            elementwise_affine=False,
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
            elementwise_affine=False,
        )
        self.mlp = EnsembleMLP(config)
