import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate
from transformers import DynamicCache

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP


class GPTLadderBlock_TP(GPTDolomiteBlock_TP):
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
