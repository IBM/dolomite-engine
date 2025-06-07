# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from .forward_diagonal import _get_autotune_configs, _rnn_forward_update


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_N"])
@triton.jit
def diagonal_rnn_varlen_forward_triton_kernel(
    x_ptr,
    x_stride_t,
    W_ptr,
    h_ptr,
    y_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    N,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_b = indices_b < B
    mask_n = indices_n < N
    mask_bn = mask_b[:, None] & mask_n[None, :]

    W = tl.load(W_ptr + indices_n, mask=mask_n)[None, :]

    if h_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(h_ptr + indices_b[:, None] * N + indices_n[None, :], mask=mask_bn)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * x_stride_t + indices_n[None, :]

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_n[None, :]

        h = _rnn_forward_update(
            h=h,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(y_ptr + indices, h, mask=mask)

        indices += x_stride_t
        start += 1


@cute_op(f"{LIBRARY_NAME}::diagonal_rnn_varlen_forward_triton", mutates_args={"output"})
def diagonal_rnn_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    B = cu_seqlens.size(0) - 1
    N = input.size(1)

    BLOCK_SIZE_N = min(1024, get_next_power_of_2(N))
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(N, meta["BLOCK_SIZE_N"]))

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        diagonal_rnn_varlen_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_t=input.stride(0),
            W_ptr=weight,
            h_ptr=input_state,
            y_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            N=N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
