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
        ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, sequence_parallel: bool, context: str
    ) -> torch.Tensor:
        if weight is not None:
            device_mesh = weight.device_mesh
            weight = dtensor_to_tensor(weight)

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

        ctx.context = context

        (_MLP_F if context == "mlp" else _ATTN_F).append(
            (x, weight, rmsnorm_denominator, is_x_1d, eps, sequence_parallel, device_mesh)
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        (_MLP_B if ctx.context == "mlp" else _ATTN_B).append(output_grad)
        return None, None, None, None, None


class _RMSNorm_Cute_B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, context: str) -> torch.Tensor:
        ctx.context = context
        return x, weight

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor, weight_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, rmsnorm_denominator, is_x_1d, eps, sequence_parallel, device_mesh = (
            _MLP_F if ctx.context == "mlp" else _ATTN_F
        ).pop()
        output_grad = (_MLP_B if ctx.context == "mlp" else _ATTN_B).pop()

        output_grad = output_grad.contiguous()

        x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)
        output_grad = wait_for_ACT(output_grad, wait_in_forward=True, wait_in_backward=False)

        x_grad, weight_grad = _backward(
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
            x_grad = x_grad.squeeze(0)

        weight_grad = tensor_to_dtensor(
            weight_grad, device_mesh=device_mesh, current_placement=Partial() if sequence_parallel else Replicate()
        )

        return x_grad, weight_grad, None, None


def rmsnorm_cute_forward(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, sequence_parallel: bool, context
) -> torch.Tensor:
    x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)
    x = _RMSNorm_Cute_F.apply(x, weight, eps, sequence_parallel, context)
    return x


def rmsnorm_cute_backward(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float | None, sequence_parallel: bool, context
) -> torch.Tensor:
    return _RMSNorm_Cute_B.apply(x, weight, eps, context)
