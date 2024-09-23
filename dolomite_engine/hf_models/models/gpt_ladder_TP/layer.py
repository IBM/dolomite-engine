import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate
from transformers import DynamicCache

from ....utils import ProcessGroupManager
from ...modeling_utils_TP import (
    dtensor_to_tensor,
    get_attention_module_TP,
    get_module_placements,
    get_normalization_function_TP,
    tensor_to_dtensor,
)
from ..gpt_dolomite_TP.mlp import MLP_TP
from ..gpt_ladder import GPTLadderConfig
from ..gpt_parallel_TP.layer import GPTParallelBlock_TP
from .linear import LadderColumnParallelLinear


class GPTLadderBlock_TP(nn.Module):
    def __init__(
        self,
        config: GPTLadderConfig,
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

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        self._patch_column_parallel(use_padding_free_transformer, sequence_parallel)
        GPTParallelBlock_TP._patch_row_parallel(self, use_padding_free_transformer, sequence_parallel)

    def forward(
        self,
        previous_attention_out: torch.Tensor,
        previous_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        residual = residual + previous_attention_out

        current_attention_out = self.ln_1(residual)

        # all gather with sequence parallel and no-op without sequence parallel
        current_attention_out = tensor_to_dtensor(
            current_attention_out, current_placement=self.placement, desired_placement=Replicate()
        )

        previous_mlp_out = dtensor_to_tensor(
            previous_mlp_out, desired_placement=self.placement, tensor_input_allowed=True
        )

        current_attention_out = self.attn(
            current_attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            current_attention_out = current_attention_out * self.m_residual

        residual = residual + previous_mlp_out

        current_mlp_out = self.ln_2(residual)

        # all gather with sequence parallel and no-op without sequence parallel
        current_mlp_out = tensor_to_dtensor(
            current_mlp_out, current_placement=self.placement, desired_placement=Replicate()
        )

        current_attention_out = dtensor_to_tensor(current_attention_out, desired_placement=self.placement)

        current_mlp_out = self.mlp(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual

    def _patch_column_parallel(self, use_padding_free_transformer: bool, sequence_parallel: bool) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        # patch to avoid multiple communication
        attn_c_attn = self.attn.c_attn
        mlp_c_fc = self.mlp.c_fc

        def _has_bias(l: nn.Linear) -> bool:
            if hasattr(l, "bias"):
                return l.bias is not None
            return False

        with torch.device("meta"):
            self.attn.c_attn = LadderColumnParallelLinear(
                in_features=attn_c_attn.in_features,
                out_features=attn_c_attn.out_features_per_device * tp_world_size,
                bias=_has_bias(attn_c_attn),
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )

            self.mlp.c_fc = LadderColumnParallelLinear(
                in_features=mlp_c_fc.in_features,
                out_features=mlp_c_fc.out_features_per_device * tp_world_size,
                bias=_has_bias(mlp_c_fc),
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )

        self.attn.c_attn.weight = attn_c_attn.weight
        self.attn.c_attn.bias = attn_c_attn.bias

        self.mlp.c_fc.weight = mlp_c_fc.weight
        self.mlp.c_fc.bias = mlp_c_fc.bias
