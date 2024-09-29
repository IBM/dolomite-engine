import torch
import torch.nn as nn
from transformers import DynamicCache

from ....utils import ProcessGroupManager, is_dtensors_enabled
from ...modeling_utils_TP import (
    dtensor_to_tensor,
    get_attention_module_TP,
    get_module_placements,
    get_normalization_function_TP,
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from ..gpt_dolomite_TP.mlp import MLP_TP
from ..gpt_parallel import GPTParallelConfig
from .linear import ParallelRowParallelLinear


class GPTParallelBlock_TP(GPTDolomiteBlock_TP):
    def __init__(
        self,
        config: GPTParallelConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        self.shared_layernorm = config.shared_layernorm

        if self.shared_layernorm:
            nn.Module.__init__(self)

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
        else:
            super().__init__(
                config=config,
                normalization_implementation=normalization_implementation,
                attention_implementation=attention_implementation,
                use_padding_free_transformer=use_padding_free_transformer,
                layer_idx=layer_idx,
                sequence_parallel=sequence_parallel,
            )

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)
        self._patch_row_parallel(use_padding_free_transformer, sequence_parallel)

        self.sequence_parallel = sequence_parallel
        self.use_padding_free_transformer = use_padding_free_transformer

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

        if self.shared_layernorm:
            hidden_states = self.ln(hidden_states)
            attention_out = self.attn(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            mlp_out = self.mlp(hidden_states)
        else:
            attention_out = self.ln_1(hidden_states)
            attention_out = self.attn(
                attention_out,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            mlp_out = self.ln_2(hidden_states)
            mlp_out = self.mlp(mlp_out)

        hidden_states = attention_out + mlp_out

        if is_dtensors_enabled():
            hidden_states = dtensor_to_tensor(hidden_states, desired_placement=self.placement)
        else:
            if self.sequence_parallel:
                hidden_states = reduce_scatter_to_sequence_parallel_region(
                    hidden_states, dim=0 if self.use_padding_free_transformer else 1
                )
            else:
                hidden_states = reduce_from_tensor_parallel_region(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states

    def _patch_row_parallel(self, use_padding_free_transformer: bool, sequence_parallel: bool) -> None:
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
