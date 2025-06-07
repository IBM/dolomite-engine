# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_math import leaky_relu, matmul, sigmoid, tanh
from ....utils import cute_op


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.jit
def _activation(x, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        x = leaky_relu(x, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        x = sigmoid(x)
    elif ACTIVATION_FUNCTION == "tanh":
        x = tanh(x)

    return x


@triton.jit
def _rnn_forward_update(h, W, x, ACTIVATION_FUNCTION, relu_negative_slope):
    h = matmul(A=h, B=W, C=x, output_dtype=x.dtype)
    h = _activation(x=h, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION, relu_negative_slope=relu_negative_slope)
    return h


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def rnn_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_s,
    W_ptr,
    W_stride_n,
    h_ptr,
    h_stride_b,
    y_ptr,
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

    indices = indices_b[:, None] * x_stride_b + pid_n * H + indices_h[None, :]

    for _ in range(S):
        h = _rnn_forward_update(
            h=h,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask_bh),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(y_ptr + indices, h, mask=mask_bh)

        indices += x_stride_s


@cute_op(f"{LIBRARY_NAME}::rnn_forward_triton", mutates_args={"output"})
def rnn_forward_triton(
    input: torch.Tensor, weight: torch.Tensor, input_state: torch.Tensor | None, output: torch.Tensor
) -> None:
    B, S, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(input.device):
        rnn_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_b=input.stride(0),
            x_stride_s=input.stride(1),
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            h_ptr=input_state,
            h_stride_b=None if input_state is None else input_state.stride(0),
            y_ptr=output,
            B=B,
            S=S,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
