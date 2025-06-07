# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp
from ....utils import cute_op
from .backward import _get_autotune_configs, _rnn_backward_update


@triton.jit
def _load_input_state(
    h_ptr,
    h_stride_b,
    pid_n,
    indices_b,
    indices_h,
    mask_bh,
    H,
    BLOCK_SIZE_B,
    BLOCK_SIZE_H,
    dtype,
):
    if h_ptr is None:
        y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)
    else:
        y_ptrs = h_ptr + indices_b[:, None] * h_stride_b + pid_n * H + indices_h[None, :]
        y_prev = tl.load(y_ptrs, mask=mask_bh)

    return y_prev


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr"])
@triton.jit
def rnn_varlen_backward_triton_kernel(
    W_ptr,
    W_stride_n,
    y_ptr,
    y_stride_t,
    h_ptr,
    h_stride_b,
    dy_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_W = pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(W_ptr + indices_W, mask=mask_hh)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    end -= 1

    indices = end * y_stride_t + pid_n * H + indices_h[None, :]
    y = tl.load(y_ptr + indices, mask=mask_bh)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for _ in range(max_seqlen - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        unfinished = end >= start
        mask = unfinished & mask_h[None, :]

        dy = tl.load(dy_ptr + indices, mask=mask) + dh

        dx_ptrs = dx_ptr + indices
        indices -= y_stride_t

        y_prev = tl.where(
            start == end,
            _load_input_state(
                h_ptr=h_ptr,
                h_stride_b=h_stride_b,
                pid_n=pid_n,
                indices_b=indices_b,
                indices_h=indices_h,
                mask_bh=mask_bh,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                dtype=W.dtype,
            ),
            tl.load(y_ptr + indices, mask=mask & (indices >= 0)),
        )

        dx, dW, dh = _rnn_backward_update(
            y=y,
            W=W,
            dy=dy,
            dW=dW,
            y_prev=y_prev,
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(dx_ptrs, dx, mask=mask)
        y = y_prev

        end -= 1

    tl.atomic_add(dW_ptr + indices_W, dW, mask=mask_hh)


@cute_op(f"{LIBRARY_NAME}::rnn_varlen_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_varlen_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
) -> None:
    _, N, H = output.size()
    B = cu_seqlens.size(0) - 1

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(output.device):
        rnn_varlen_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            y_ptr=output,
            y_stride_t=output.stride(0),
            h_ptr=input_state,
            h_stride_b=None if input_state is None else input_state.stride(0),
            dy_ptr=output_grad,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
