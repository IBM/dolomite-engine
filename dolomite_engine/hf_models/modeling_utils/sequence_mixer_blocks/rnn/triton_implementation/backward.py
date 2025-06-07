# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_math import clamp, leaky_relu_backward, matmul, sigmoid_backward, tanh_backward
from ....utils import cute_op
from .forward import _get_autotune_configs


@triton.jit
def _activation_backward(y, dy, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        dy *= leaky_relu_backward(y, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        dy *= sigmoid_backward(y)
    elif ACTIVATION_FUNCTION == "tanh":
        dy *= tanh_backward(y)

    return dy


@triton.jit
def _rnn_backward_update(y, W, dy, dW, y_prev, ACTIVATION_FUNCTION: tl.constexpr, relu_negative_slope):
    dx = _activation_backward(
        y=y, dy=dy, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope
    )

    dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
    dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

    return dx, dW, dh


@triton.jit
def _load_previous_output(
    h_ptr,
    h_stride_b,
    y_ptrs,
    pid_n,
    H,
    indices_b,
    indices_h,
    mask_bh,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    s,
    dtype,
):
    if s == 0:
        if h_ptr is None:
            y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)
        else:
            y_prev = tl.load(h_ptr + indices_b[:, None] * h_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh)
    else:
        y_prev = tl.load(y_ptrs, mask=mask_bh)

    return y_prev


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr"])
@triton.jit
def rnn_backward_triton_kernel(
    W_ptr,
    W_stride_n,
    y_ptr,
    y_stride_b,
    y_stride_s,
    h_ptr,
    h_stride_b,
    dy_ptr,
    dx_ptr,
    dW_ptr,
    gradient_clipping,
    B,
    S,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)
    indices_weight = pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :]

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(W_ptr + indices_weight, mask=mask_hh)

    indices = indices_b[:, None] * y_stride_b + (S - 1) * y_stride_s + pid_n * H + indices_h[None, :]
    y = tl.load(y_ptr + indices, mask=mask_bh)

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dy = tl.load(dy_ptr + indices, mask=mask_bh) + dh

        dx_ptrs = dx_ptr + indices
        indices -= y_stride_s

        y_prev = _load_previous_output(
            h_ptr=h_ptr,
            h_stride_b=h_stride_b,
            y_ptrs=y_ptr + indices,
            pid_n=pid_n,
            H=H,
            indices_b=indices_b,
            indices_h=indices_h,
            mask_bh=mask_bh,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            s=s,
            dtype=W.dtype,
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

        tl.store(dx_ptrs, dx, mask=mask_bh)
        y = y_prev

    tl.atomic_add(dW_ptr + indices_weight, dW, mask=mask_hh)


@cute_op(f"{LIBRARY_NAME}::rnn_backward_triton", mutates_args={"input_grad", "weight_grad"})
def rnn_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    gradient_clipping: float | None,
) -> None:
    B, S, N, H = output.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        rnn_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            y_ptr=output,
            y_stride_b=output.stride(0),
            y_stride_s=output.stride(1),
            h_ptr=input_state,
            h_stride_b=None if input_state is None else input_state.stride(0),
            dy_ptr=output_grad,
            dx_ptr=input_grad,
            dW_ptr=weight_grad,
            gradient_clipping=gradient_clipping,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
