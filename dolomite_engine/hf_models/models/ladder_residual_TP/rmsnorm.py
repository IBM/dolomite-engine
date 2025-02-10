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
        context,
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

        output, rmsnorm_denominator = _forward(x=x, weight=weight, eps=eps, memory_efficient=False)

        if is_x_1d:
            output = output.squeeze(0)

        context.x = x
        context.weight = weight
        context.rmsnorm_denominator = rmsnorm_denominator
        context.is_x_1d = is_x_1d
        context.eps = eps

        return output, x, weight

    @staticmethod
    @ensure_contiguous
    def backward(
        ctx, output_grad: torch.Tensor, x_grad: torch.Tensor, weight_grad: torch.Tensor
    ) -> tuple[torch.Tensor | None]:
        return x_grad, weight_grad, None, None


class _RMSNorm_Cute_Backward(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        output: torch.Tensor,
        context,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, context.rmsnorm_denominator)
        ctx.eps = context.eps
        ctx.is_x_1d = context.is_x_1d

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
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, output_grad, None


def rmsnorm_cute_forward_only(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    context,
) -> torch.Tensor:
    return _RMSNorm_Cute_Forward.apply(x, weight, eps, context)


def rmsnorm_cute_backward_only(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    context,
) -> torch.Tensor:
    return _RMSNorm_Cute_Backward.apply(x, weight, output, context)
