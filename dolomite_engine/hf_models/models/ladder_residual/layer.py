import torch
import torch.nn.functional as F
from cute_kernels import CutoTuneParameter
from cute_kernels.kernels.rmsnorm.backward import rmsnorm_backward_triton
from cute_kernels.kernels.rmsnorm.forward import _forward as rmsnorm_forward
from cute_kernels.kernels.swiglu_unchunked.forward import _forward as swiglu_unchunked_forward
from cute_kernels.math import get_next_power_of_2
from transformers import DynamicCache

from ..gpt_dolomite.layer import GPTDolomiteBlock


def _mlp_forward(
    x: torch.Tensor,
    c_fc_weight: torch.Tensor,
    c_fc_bias: torch.Tensor,
    c_proj_weight: torch.Tensor,
    c_proj_bias: torch.Tensor,
) -> tuple[torch.Tensor]:
    assert c_fc_bias is None
    assert c_proj_bias is None

    c_fc_out = F.linear(x, c_fc_weight, c_fc_bias)

    swiglu_output = swiglu_unchunked_forward(
        c_fc_out,
        kernel_backend=CutoTuneParameter(),
        BLOCK_SIZE_B=CutoTuneParameter(),
        BLOCK_SIZE_H=CutoTuneParameter(),
    )

    c_proj_out = F.linear(swiglu_output, c_proj_weight, c_proj_bias)

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
        mlp0_c_fc_bias: torch.Tensor,
        mlp0_c_proj_weight: torch.Tensor,
        mlp0_c_proj_bias: torch.Tensor,
        mlp_c_fc_weight: torch.Tensor,
        mlp_c_fc_bias: torch.Tensor,
        mlp_c_proj_weight: torch.Tensor,
        mlp_c_proj_bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor]:
        if current_attention_out is not None:
            residual = residual + current_attention_out

        attention_input, attention_rmsnorm_denominator = rmsnorm_forward(
            x=residual,
            weight=ln_1_weight,
            eps=eps,
            memory_efficient=False,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        attention_c_fc_out, attention_swiglu_out, attention_c_proj_out = _mlp_forward(
            x=attention_input,
            c_fc_weight=mlp0_c_fc_weight,
            c_fc_bias=mlp0_c_fc_bias,
            c_proj_weight=mlp0_c_proj_weight,
            c_proj_bias=mlp0_c_proj_bias,
        )

        if current_mlp_out is not None:
            residual = residual + current_mlp_out

        mlp_input, mlp_rmsnorm_denominator = rmsnorm_forward(
            x=residual,
            weight=ln_2_weight,
            eps=eps,
            memory_efficient=False,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        c_fc_out, swiglu_out, c_proj_out = _mlp_forward(
            x=mlp_input,
            c_fc_weight=mlp_c_fc_weight,
            c_fc_bias=mlp_c_fc_bias,
            c_proj_weight=mlp_c_proj_weight,
            c_proj_bias=mlp_c_proj_bias,
        )

        ctx.save_for_backward(
            attention_rmsnorm_denominator,
            attention_c_fc_out,
            attention_swiglu_out,
            attention_c_proj_out,
            ln_1_weight,
            mlp0_c_fc_weight,
            mlp0_c_fc_bias,
            mlp0_c_proj_weight,
            mlp0_c_proj_bias,
        )

        ctx.eps = eps

        return attention_c_proj_out, c_proj_out, residual


class LadderResidualBlock(GPTDolomiteBlock):
    def forward(
        self,
        current_attention_out: torch.Tensor | None,
        current_mlp_out: torch.Tensor | None,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if current_attention_out is not None:
            residual = residual + current_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self._sequence_mixer_forward(
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

        current_mlp_out = self.ln_2(residual)
        current_mlp_out = self.mlp_block(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual
