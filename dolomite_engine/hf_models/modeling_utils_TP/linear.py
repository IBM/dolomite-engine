import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from ...utils import ProcessGroupManager
from ..modeling_utils import ParameterizedLinear
from ..utils import divide_if_divisible
from .dtensor_module import DTensorModule
from .TP import (
    all_gather_from_sequence_parallel_region,
    copy_to_tensor_parallel_region,
    dtensor_to_tensor,
    get_module_placements,
    tensor_to_dtensor,
    use_async_tensor_parallel,
)


class ReplicatedLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, std)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
                )
            )

        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        if use_async_tensor_parallel():
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=self.input_placement)
        input = super().forward(input)
        input = dtensor_to_tensor(
            input, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
        )
        return input


class ColumnParallelLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.out_features_per_device = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            in_features=in_features,
            out_features=self.out_features_per_device,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
                )
            )

        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel
        self.use_async_tensor_parallel = use_async_tensor_parallel()

        if self.use_async_tensor_parallel:
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # FIXME dtensor redistribute uses alltoall for large number of GPUs
        if self.use_async_tensor_parallel:
            if self.sequence_parallel:
                input = all_gather_from_sequence_parallel_region(
                    input, dim=0 if self.use_padding_free_transformer else 1
                )
            else:
                input = copy_to_tensor_parallel_region(input)

            input = F.linear(input, self.weight.to_local(), None if self.bias is None else self.bias.to_local())
        else:
            input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=self.input_placement)
            input = super().forward(input)
            input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=Shard(-1))

        return input

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )


class RowParallelLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.in_features_per_device = divide_if_divisible(
            in_features,
            self.tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        super().__init__(
            in_features=self.in_features_per_device,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(1)]
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
                )
            )

        self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        if use_async_tensor_parallel():
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=Shard(-1))
        input = super().forward(input)
        input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.output_placement)
        return input

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.bias is not None
        )
