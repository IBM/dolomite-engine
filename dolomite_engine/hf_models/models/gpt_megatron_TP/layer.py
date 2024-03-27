import torch.nn as nn

from ...enums import AttentionHeadType
from ...modeling_utils import get_normalization_function
from ...modeling_utils_TP import get_attention_module
from ...safetensors import SafeTensorsWeightsManager
from ..gpt_megatron import GPTMegatronConfig
from ..gpt_megatron.layer import GPTMegatronBlock
from .mlp import MLP_TP


class GPTMegatronBlock_TP(GPTMegatronBlock):
    def __init__(
        self,
        config: GPTMegatronConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layer_idx = layer_idx

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = get_attention_module(
            config,
            True,
            attention_implementation,
            use_padding_free_transformer,
            layer_idx,
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = MLP_TP(
            config.hidden_size,
            self.inner_dim,
            config.activation_function,
            config.add_bias,
            config.resid_pdrop,
        )

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        state_dict = {
            "weight": safetensors_weight_manager.get_tensor(prefix + "ln_1.weight"),
            "bias": safetensors_weight_manager.get_tensor(prefix + "ln_1.bias"),
        }
        self.ln_1.load_state_dict(state_dict)

        state_dict = {
            "weight": safetensors_weight_manager.get_tensor(prefix + "ln_2.weight"),
            "bias": safetensors_weight_manager.get_tensor(prefix + "ln_2.bias"),
        }
        self.ln_2.load_state_dict(state_dict)

        self.attn.load_unsharded_weights(safetensors_weight_manager, prefix + "attn.")
        self.mlp.load_unsharded_weights(safetensors_weight_manager, prefix + "mlp.")
