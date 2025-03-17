import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Partial, Replicate

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from ...enums import Kernel
from ...kernels import is_kernel_allowed, wait_for_ACT
from ...utils import ProcessGroupManager, is_cute_kernels_available
from .dtensor_module import DTensorModule
from .TP import get_module_placements


if is_cute_kernels_available():
    from cute_kernels import rmsnorm_cute


class LayerNorm_TP(nn.LayerNorm, DTensorModule):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                current_placement=Replicate(),
            )
        )
        self.bias = nn.Parameter(
            tensor_to_dtensor(
                self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Replicate()
            )
        )

        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = tensor_to_dtensor(hidden_states, device_mesh=self.tp_mesh, current_placement=self.placement)
        hidden_states = super().forward(hidden_states)
        hidden_states = dtensor_to_tensor(hidden_states, device_mesh=self.tp_mesh, desired_placement=self.placement)
        return hidden_states


class RMSNorm_TP(nn.RMSNorm, DTensorModule):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Replicate()
            )
        )

        self.sequence_parallel = sequence_parallel
        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_kernel_allowed(Kernel.rmsnorm_cute):
            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=True, wait_in_backward=False)
            hidden_states = rmsnorm_cute(
                x=hidden_states,
                weight=dtensor_to_tensor(
                    self.weight, grad_placement=Partial() if self.sequence_parallel else Replicate()
                ),
                eps=self.eps,
                memory_efficient=False,
            )
            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
        else:
            hidden_states = tensor_to_dtensor(
                hidden_states, device_mesh=self.tp_mesh, current_placement=self.placement
            )
            hidden_states = super().forward(hidden_states)
            hidden_states = dtensor_to_tensor(
                hidden_states, device_mesh=self.tp_mesh, desired_placement=self.placement
            )

        return hidden_states


_NORMALIZATION_FUNCTIONS = {
    "layernorm": LayerNorm_TP,
    "rmsnorm": RMSNorm_TP,
}


def get_normalization_function_TP(
    normalization_function: str,
    normalized_shape: int,
    eps: float = 1e-5,
    use_padding_free_transformer: bool = False,
    sequence_parallel: bool = False,
) -> nn.LayerNorm:
    if normalization_function in _NORMALIZATION_FUNCTIONS:
        normalization = _NORMALIZATION_FUNCTIONS[normalization_function](
            normalized_shape,
            eps=eps,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
    else:
        raise ValueError(f"unexpected `normalization_function` {normalization_function}")

    return normalization
