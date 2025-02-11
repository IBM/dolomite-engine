import torch
from cute_kernels.cutotune import CutoTuneParameter
from cute_kernels.kernels.rmsnorm.backward import _backward
from cute_kernels.kernels.rmsnorm.forward import _forward
from cute_kernels.utils import ensure_contiguous


_MLP_F = []
_MLP_B = []


class _RMSNorm_Cute_F(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        x_input = x.unsqueeze(0) if is_x_1d else x

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        output, rmsnorm_denominator = _forward(
            x=x_input,
            weight=weight,
            eps=eps,
            memory_efficient=False,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        if is_x_1d:
            output = output.squeeze(0)

        _MLP_F.append((x, weight, rmsnorm_denominator, is_x_1d, eps))

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        _MLP_B.append(output_grad)
        return None, None, None


class _RMSNorm_Cute_B(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
        return x, weight

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator, is_x_1d, eps = _MLP_F.pop()
        output_grad = _MLP_B.pop()

        x_grad, weight_grad = _backward(
            x=x,
            weight=weight,
            eps=ctx.eps,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, None


def rmsnorm_cute_forward(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    return _RMSNorm_Cute_F.apply(x, weight, eps)


def rmsnorm_cute_backward(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    return _RMSNorm_Cute_B.apply(x, weight, eps)
