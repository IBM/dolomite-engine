import torch
from cute_kernels import rmsnorm_cute
from torch.distributed._tensor.placement_types import Partial, Replicate
from transformers import DynamicCache

from ....distributed import dtensor_to_tensor
from ....kernels import wait_for_ACT
from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from .rmsnorm import rmsnorm_cute_backward, rmsnorm_cute_forward


def rmsnorm_cute_wrapper(
    input: torch.Tensor, weight: torch.Tensor, eps: float, sequence_parallel: bool
) -> torch.Tensor:
    input = wait_for_ACT(input, wait_in_forward=True, wait_in_backward=False)
    input = rmsnorm_cute(
        x=input,
        weight=dtensor_to_tensor(weight, grad_placement=Partial() if sequence_parallel else Replicate()),
        eps=eps,
    )
    input = wait_for_ACT(input, wait_in_forward=False, wait_in_backward=True)
    return input


class LadderResidualBlock_TP(GPTDolomiteBlock_TP):
    def __init__(
        self, config, attention_implementation, use_padding_free_transformer, layer_idx=None, sequence_parallel=False
    ):
        super().__init__(config, attention_implementation, use_padding_free_transformer, layer_idx, sequence_parallel)
        self.sequence_parallel = sequence_parallel

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
            residual = residual + current_attention_out

        ln_2_weight = self.ln_2.weight

        current_attention_out = rmsnorm_cute_wrapper(residual, self.ln_1.weight, self.ln_1.eps, self.sequence_parallel)
        current_attention_out = self.sequence_mixer(
            current_attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            current_attention_out = current_attention_out * self.m_residual

        if current_mlp_out is not None:
            residual = residual + current_mlp_out

        # current_mlp_out = rmsnorm_cute_wrapper(residual, self.ln_2.weight, self.ln_2.eps, self.sequence_parallel)
        current_mlp_out = rmsnorm_cute_forward(residual, ln_2_weight, self.ln_2.eps, self.sequence_parallel)
        current_mlp_out = wait_for_ACT(current_mlp_out, wait_in_forward=False, wait_in_backward=True)

        current_mlp_out = self.mlp_block(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual
