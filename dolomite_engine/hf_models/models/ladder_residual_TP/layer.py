import torch
import torch.nn.functional as F
from cute_kernels import CutoTuneParameter
from cute_kernels.kernels.rmsnorm.backward import rmsnorm_backward_triton
from cute_kernels.kernels.rmsnorm.forward import _forward as rmsnorm_forward
from cute_kernels.kernels.swiglu_unchunked.forward import _forward as swiglu_unchunked_forward
from transformers import DynamicCache

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...modeling_utils_TP import get_mlp_block_TP
from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from ..ladder_residual.layer import LadderResidualBlock


def _mlp_forward(x: torch.Tensor, c_fc_weight: torch.Tensor, c_proj_weight: torch.Tensor) -> tuple[torch.Tensor]:
    c_fc_out = F.linear(x, c_fc_weight)

    swiglu_output = swiglu_unchunked_forward(
        c_fc_out,
        kernel_backend=CutoTuneParameter(),
        BLOCK_SIZE_B=CutoTuneParameter(),
        BLOCK_SIZE_H=CutoTuneParameter(),
    )

    c_proj_out = F.linear(swiglu_output, c_proj_weight)

    return c_fc_out, swiglu_output, c_proj_out


class _OverlappableBlock(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_attention_out: torch.Tensor | None,
        current_mlp_out: torch.Tensor | None,
        residual: torch.Tensor,
        ln_1_weight: torch.Tensor,
        ln_2_weight: torch.Tensor,
        mlp0_c_fc_weight: torch.Tensor,
        mlp0_c_proj_weight: torch.Tensor,
        mlp_c_fc_weight: torch.Tensor,
        mlp_c_proj_weight: torch.Tensor,
        eps1: float,
        eps2: float,
    ) -> tuple[torch.Tensor]:
        if current_attention_out is None:
            attention_rmsnorm_input = residual
        else:
            attention_rmsnorm_input = residual + current_attention_out

        attention_input, attention_rmsnorm_denominator = rmsnorm_forward(
            x=attention_rmsnorm_input,
            weight=ln_1_weight,
            eps=eps1,
            memory_efficient=False,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        attention_c_fc_out, attention_swiglu_out, attention_c_proj_out = _mlp_forward(
            x=attention_input, c_fc_weight=mlp0_c_fc_weight, c_proj_weight=mlp0_c_proj_weight
        )

        if current_mlp_out is None:
            mlp_rmsnorm_input = attention_rmsnorm_input
        else:
            mlp_rmsnorm_input = attention_rmsnorm_input + current_mlp_out

        mlp_input, mlp_rmsnorm_denominator = rmsnorm_forward(
            x=mlp_rmsnorm_input,
            weight=ln_2_weight,
            eps=eps2,
            memory_efficient=False,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        mlp_c_fc_out, mlp_swiglu_out, mlp_c_proj_out = _mlp_forward(
            x=mlp_input, c_fc_weight=mlp_c_fc_weight, c_proj_weight=mlp_c_proj_weight
        )

        ctx.save_for_backward(
            attention_rmsnorm_denominator,
            attention_c_fc_out,
            attention_swiglu_out,
            attention_c_proj_out,
            mlp_rmsnorm_denominator,
            mlp_c_fc_out,
            mlp_swiglu_out,
            mlp_c_proj_out,
            ln_1_weight,
            ln_2_weight,
            mlp0_c_fc_weight,
            mlp0_c_proj_weight,
            mlp_c_fc_weight,
            mlp_c_proj_weight,
        )

        ctx.eps1 = eps1
        ctx.eps2 = eps2

        return attention_c_proj_out, mlp_c_proj_out, mlp_rmsnorm_input

    @staticmethod
    def backward(
        ctx,
        attention_c_proj_out_grad: torch.Tensor,
        mlp_c_proj_out_grad: torch.Tensor,
        mlp_rmsnorm_input_grad: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (
            attention_c_proj_out_grad,
            mlp_c_proj_out_grad,
            mlp_rmsnorm_input_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LadderResidualBlock_TP(GPTDolomiteBlock_TP):
    def __init__(self, config, use_padding_free_transformer, layer_idx=None, sequence_parallel=False):
        super().__init__(config, use_padding_free_transformer, layer_idx, sequence_parallel)

        self.mlp0_block = get_mlp_block_TP(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )

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
        if is_kernel_allowed(Kernel.ladder_residual_overlapped_layer):
            assert self.m_residual in [None, 1]
            assert self.ln_1.eps == self.ln_2.eps

            current_attention_out, current_mlp_out, residual = _OverlappableBlock.apply(
                current_attention_out,
                current_mlp_out,
                residual,
                self.ln_1.weight,
                self.ln_2.weight,
                self.mlp0_block.c_fc.weight,
                self.mlp0_block.c_proj.weight,
                self.mlp_block.c_fc.weight,
                self.mlp_block.c_proj.weight,
                self.ln_1.eps,
                self.ln_2.eps,
            )
        else:
            current_attention_out, current_mlp_out, residual = LadderResidualBlock.forward(
                self,
                current_attention_out=current_attention_out,
                current_mlp_out=current_mlp_out,
                residual=residual,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return current_attention_out, current_mlp_out, residual
