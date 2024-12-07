import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Partial, Replicate
from transformers import DynamicCache

from ....distributed import ProcessGroupManager
from ....utils import ProcessGroupManager
from ...modeling_utils_TP import get_normalization_function_TP
from ..ladder_residual import LadderResidualConfig
from .attention import get_attention_module_TP
from .mlp import LadderMLP_TP
from .redistribute import backward_redistribute, forward_redistribute


class LadderResidualBlock_TP(nn.Module):
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

    def forward(
        self,
        current_attention_out: torch.Tensor,
        current_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if current_attention_out is not None:
            if self.m_residual is not None:
                current_attention_out = current_attention_out * self.m_residual

            residual = residual + current_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self.attn(
            current_attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if current_mlp_out is not None:
            if self.m_residual is not None:
                current_mlp_out = current_mlp_out * self.m_residual

            residual = residual + current_mlp_out

        current_mlp_out = self.ln_2(residual)
        current_mlp_out = backward_redistribute(
            current_mlp_out,
            device_mesh=self.tp_mesh,
            current_backward_placement=Partial(),
            desired_backward_placement=Replicate(),
        )
        current_attention_out, current_mlp_out = self.mlp(current_attention_out, current_mlp_out)
        current_mlp_out = forward_redistribute(
            current_mlp_out, device_mesh=self.tp_mesh, current_placement=Partial(), desired_placement=Replicate()
        )

        return current_attention_out, current_mlp_out, residual
