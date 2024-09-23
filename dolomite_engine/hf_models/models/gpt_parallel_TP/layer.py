import torch
import torch.nn as nn
from transformers import DynamicCache

from dolomite_engine.hf_models.models.gpt_dolomite.config import GPTDolomiteConfig

from ....utils import ProcessGroupManager
from ...modeling_utils_TP import dtensor_to_tensor, get_attention_module_TP, get_module_placements
from ..gpt_dolomite_TP.mlp import MLP_TP
from .linear import ParallelRowParallelLinear
from .normalization import get_normalization_function_TP


class GPTParallelBlock_TP(nn.Module):
    def __init__(
        self,
        config: GPTDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln = get_normalization_function_TP(
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
        self.mlp = MLP_TP(
            config, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        # patch to avoid multiple communication
        attn_c_proj = self.attn.c_proj
        mlp_c_proj = self.mlp.c_proj

        def _has_bias(l: nn.Linear) -> bool:
            if hasattr(l, "bias"):
                return l.bias is not None
            return False

        with torch.device("meta"):
            self.attn.c_proj = ParallelRowParallelLinear(
                in_features=attn_c_proj.in_features_per_device * tp_world_size,
                out_features=attn_c_proj.out_features,
                bias=_has_bias(attn_c_proj),
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )

            self.mlp.c_proj = ParallelRowParallelLinear(
                in_features=mlp_c_proj.in_features_per_device * tp_world_size,
                out_features=mlp_c_proj.out_features,
                bias=_has_bias(mlp_c_proj),
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )

        self.attn.c_proj.weight = attn_c_proj.weight
        self.attn.c_proj.bias = attn_c_proj.bias

        self.mlp.c_proj.weight = mlp_c_proj.weight
        self.mlp.c_proj.bias = mlp_c_proj.bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states

        hidden_states = self.ln(hidden_states)
        attention_out, mlp_out = hidden_states.chunk(2)

        attention_out = attention_out.squeeze(0)
        mlp_out = mlp_out.squeeze(0)

        attention_out = self.attn(
            attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        mlp_out = self.mlp(mlp_out)

        hidden_states = attention_out + mlp_out

        hidden_states = dtensor_to_tensor(hidden_states, desired_placement=self.placement)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states
