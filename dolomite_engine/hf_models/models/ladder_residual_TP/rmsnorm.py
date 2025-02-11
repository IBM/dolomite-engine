import torch
from cute_kernels.cutotune import CutoTuneParameter
from cute_kernels.kernels.rmsnorm.backward import _backward
from cute_kernels.kernels.rmsnorm.forward import _forward
from cute_kernels.utils import ensure_contiguous
from torch.distributed._tensor.placement_types import Partial, Replicate

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ....kernels import wait_for_ACT


_ATTN_F = []
_ATTN_B = []
_MLP_F = []
_MLP_B = []


class _RMSNorm_Cute_F(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        residual: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        sequence_parallel: bool,
        context: str,
    ) -> torch.Tensor:
        if weight is not None:
            device_mesh = weight.device_mesh
            weight = dtensor_to_tensor(weight)

            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == residual.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == residual.type(), "tensors weight and y should have same dtype"

        is_x_1d = residual.dim() == 1
        x_input = residual.unsqueeze(0) if is_x_1d else residual

        if eps is None:
            eps = torch.finfo(residual.dtype).eps

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

        ctx.context = context

        (_MLP_F if context == "mlp" else _ATTN_F).append(
            (residual, weight, rmsnorm_denominator, is_x_1d, eps, sequence_parallel, device_mesh)
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        (_MLP_B if ctx.context == "mlp" else _ATTN_B).append(output_grad)
        return None, None, None, None, None


class _RMSNorm_Cute_B(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, residual: torch.Tensor, weight: torch.Tensor | None, eps: float | None, context: str
    ) -> torch.Tensor:
        ctx.context = context
        return residual, weight

    @staticmethod
    def backward(ctx, residual_grad: torch.Tensor, weight_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        residual, weight, rmsnorm_denominator, is_x_1d, eps, sequence_parallel, device_mesh = (
            _MLP_F if ctx.context == "mlp" else _ATTN_F
        ).pop()
        output_grad = (_MLP_B if ctx.context == "mlp" else _ATTN_B).pop()

        output_grad = output_grad.contiguous()

        x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)
        output_grad = wait_for_ACT(output_grad, wait_in_forward=True, wait_in_backward=False)

        residual_grad, weight_grad = _backward(
            x=x,
            weight=weight,
            eps=eps,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            kernel_backend=CutoTuneParameter(),
            BLOCK_SIZE_B=CutoTuneParameter(),
            BLOCK_SIZE_H=CutoTuneParameter(),
        )

        if is_x_1d:
            residual = residual.squeeze(0)

        weight_grad = tensor_to_dtensor(
            weight_grad, device_mesh=device_mesh, current_placement=Partial() if sequence_parallel else Replicate()
        )

        return residual, weight_grad, None, None


def rmsnorm_cute_forward(
    residual: torch.Tensor, weight: torch.Tensor | None, eps: float | None, sequence_parallel: bool, context
) -> torch.Tensor:
    residual = wait_for_ACT(residual, wait_in_forward=True, wait_in_backward=False)
    residual = _RMSNorm_Cute_F.apply(residual, weight, eps, sequence_parallel, context)
    return residual


def rmsnorm_cute_backward(
    residual: torch.Tensor, weight: torch.Tensor | None, eps: float | None, sequence_parallel: bool, context
) -> torch.Tensor:
    return _RMSNorm_Cute_B.apply(residual, weight, eps, context)
