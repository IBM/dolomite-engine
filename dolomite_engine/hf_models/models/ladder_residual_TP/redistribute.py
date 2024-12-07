import torch
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import DeviceMesh

from ....distributed import dtensor_to_tensor, tensor_to_dtensor


class _ForwardRedistribute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, device_mesh: DeviceMesh, current_placement: Placement, desired_placement: Placement
    ) -> torch.Tensor:
        x = tensor_to_dtensor(x, device_mesh=device_mesh, current_placement=current_placement)
        x = dtensor_to_tensor(x, device_mesh=device_mesh, desired_placement=desired_placement)
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        return x_grad, None, None, None


class _BackwardRedistribute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        device_mesh: DeviceMesh,
        current_backward_placement: Placement,
        desired_backward_placement: Placement,
    ) -> torch.Tensor:
        ctx.device_mesh = device_mesh
        ctx.current_backward_placement = current_backward_placement
        ctx.desired_backward_placement = desired_backward_placement
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> torch.Tensor:
        device_mesh = ctx.device_mesh
        current_placement = ctx.current_backward_placement
        desired_placement = ctx.desired_backward_placement

        x_grad = tensor_to_dtensor(x_grad, device_mesh=device_mesh, current_placement=current_placement)
        x_grad = dtensor_to_tensor(x_grad, device_mesh=device_mesh, desired_placement=desired_placement)
        return x_grad, None, None, None


def forward_redistribute(
    x: torch.Tensor, device_mesh: DeviceMesh, current_placement: Placement, desired_placement: Placement
) -> torch.Tensor:
    return _ForwardRedistribute.apply(x, device_mesh, current_placement, desired_placement)


def backward_redistribute(
    x: torch.Tensor,
    device_mesh: DeviceMesh,
    current_backward_placement: Placement,
    desired_backward_placement: Placement,
) -> torch.Tensor:
    return _BackwardRedistribute.apply(x, device_mesh, current_backward_placement, desired_backward_placement)
