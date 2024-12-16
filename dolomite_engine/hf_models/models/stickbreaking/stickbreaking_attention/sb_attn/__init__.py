import math

import torch
import triton.language as tl

from .sb_bwd import _bwd
from .sb_fwd import _fwd


FWD_BLOCK_M: tl.constexpr = 64
FWD_BLOCK_N: tl.constexpr = 32
BWD_BLOCK_M: tl.constexpr = 64
BWD_BLOCK_N: tl.constexpr = 32


class StickBreakingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inv_temp: float):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        BLOCK_M = FWD_BLOCK_M
        BLOCK_N = FWD_BLOCK_N
        o, rem, neg_log_acc = _fwd(
            q, k, v, logit_scale=inv_temp, no_grad=no_grad, return_attention=False, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(q, k, v, neg_log_acc)
        ctx.logit_scale = logit_scale
        return o, rem

    @staticmethod
    def backward(ctx, do: torch.Tensor, drem: torch.Tensor):
        logit_scale = ctx.logit_scale
        q, k, v, neg_log_acc = ctx.saved_tensors
        BLOCK_M = BWD_BLOCK_M
        BLOCK_N = BWD_BLOCK_N
        dq, dk, dv = _bwd(
            do,
            drem,
            q,
            k,
            v,
            neg_log_acc,
            logit_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return dq, dk, dv, None


def sb_attn(q, k, v, inv_temp=None, zero_start=True):
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))
    return sb_attn_(q, k, v, inv_temp)


def sb_attn_(q, k, v, inv_temp):
    return StickBreakingAttention.apply(q, k, v, inv_temp)
