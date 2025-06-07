# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from .forward import _get_autotune_configs, _rnn_forward_update


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def rnn_varlen_forward_triton_kernel(
    x_ptr,
    x_stride_t,
    W_ptr,
    W_stride_n,
    h_ptr,
    h_stride_b,
    y_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    W = tl.load(
        W_ptr + pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :],
        mask=mask_h[:, None] & mask_h[None, :],
    )

    if h_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(h_ptr + indices_b[:, None] * h_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * x_stride_t + pid_n * H + indices_h[None, :]

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_h[None, :]

        h = _rnn_forward_update(
            h=h,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask_bh),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(y_ptr + indices, h, mask=mask)

        indices += x_stride_t
        start += 1


@cute_op(f"{LIBRARY_NAME}::rnn_varlen_forward_triton", mutates_args={"output"})
def rnn_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    B = cu_seqlens.size(0) - 1
    _, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        rnn_varlen_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_t=input.stride(0),
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            h_ptr=input_state,
            h_stride_b=None if input_state is None else input_state.stride(0),
            y_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
