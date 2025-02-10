import torch
from cute_kernels.cutotune import CutoTuneParameter
from cute_kernels.enums import KernelBackend
from cute_kernels.kernels.rmsnorm.backward import _backward
from cute_kernels.kernels.rmsnorm.forward import _forward
from cute_kernels.utils import ensure_contiguous


class _RMSNorm_Cute_Forward(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        memory_efficient: bool,
        kernel_backend_forward: KernelBackend,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_H_forward: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        output, rmsnorm_denominator = _forward(
            x=x,
            weight=weight,
            eps=eps,
            memory_efficient=memory_efficient,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        if is_x_1d:
            output = output.squeeze(0)

        return output, x, weight, rmsnorm_denominator, is_x_1d

    @staticmethod
    @ensure_contiguous
    def backward(
        ctx,
        output_grad: torch.Tensor,
        x_grad: torch.Tensor,
        weight_grad: torch.Tensor | None,
        rmsnorm_denominator: torch.Tensor | None,
        is_x_1d_grad: bool,
    ) -> tuple[torch.Tensor | None]:
        return x_grad, weight_grad, *[None] * 5


class _RMSNorm_Cute_Backward(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        output: torch.Tensor,
        rmsnorm_denominator: torch.Tensor | None,
        eps: float | None,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_backward: int,
        is_x_1d: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.is_x_1d = is_x_1d
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator = ctx.saved_tensors

        x_grad, weight_grad = _backward(
            x=x,
            weight=weight,
            eps=ctx.eps,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, *[None] * 7


def rmsnorm_cute_forward_only(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    memory_efficient: bool = False,
    kernel_backend_forward: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_Cute_Forward.apply(
        x, weight, eps, memory_efficient, kernel_backend_forward, BLOCK_SIZE_B_forward, BLOCK_SIZE_H_forward
    )


def rmsnorm_cute_backward_only(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    rmsnorm_denominator: torch.Tensor | None,
    eps: float | None,
    kernel_backend_backward: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
    is_x_1d: bool = False,
) -> torch.Tensor:
    return _RMSNorm_Cute_Backward.apply(
        x,
        weight,
        output,
        rmsnorm_denominator,
        eps,
        kernel_backend_backward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_backward,
        is_x_1d,
    )
