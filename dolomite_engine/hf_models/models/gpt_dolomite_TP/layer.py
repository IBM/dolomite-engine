import torch.nn as nn

from ....utils import SafeTensorsWeightsManager
from ...enums import AttentionHeadType
from ...modeling_utils_TP import get_attention_module_TP, get_normalization_function_TP
from ..gpt_dolomite import GPTDolomiteConfig
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .mlp import MLP_TP


class GPTDolomiteBlock_TP(GPTDolomiteBlock):
    def __init__(
        self,
        config: GPTDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
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
        self.mlp = MLP_TP(
            config, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_1.weight")}
        if hasattr(self.ln_1, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_1.bias")
        self.ln_1.load_state_dict(state_dict)

        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_2.weight")}
        if hasattr(self.ln_2, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_2.bias")
        self.ln_2.load_state_dict(state_dict)

        self.attn.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "attn.")
        self.mlp.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "mlp.")
