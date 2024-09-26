import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ReduceOp
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from ...utils import ProcessGroupManager, is_dtensors_enabled
from ..modeling_utils import ParameterizedLinear
from ..utils import divide_if_divisible
from .dtensor_module import DTensorModule
from .TP import (
    copy_to_tensor_parallel_region,
    dtensor_to_tensor,
    get_module_placements,
    reduce_from_tensor_parallel_region,
    tensor_to_dtensor,
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

        if sequence_parallel:
            if use_padding_free_transformer:
                self.input_placement = Shard(0)
            else:
                self.input_placement = Shard(1)
        else:
            self.input_placement = Replicate()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=self.input_placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=Replicate(), grad_placement=Partial())
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

        self.sequence_parallel = sequence_parallel
        self.use_padding_free_transformer = use_padding_free_transformer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_dtensors_enabled():
            input = tensor_to_dtensor(input, current_placement=self.input_placement)
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=Shard(-1))
        else:
            assert not self.use_padding_free_transformer
            assert not self.sequence_parallel

            input = copy_to_tensor_parallel_region(input, op=ReduceOp.SUM)
            input = F.linear(input, self.weight.to_local(), None if self.bias is None else self.bias.to_local())

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

        self.sequence_parallel = sequence_parallel

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

        self.sequence_parallel = sequence_parallel
        self.use_padding_free_transformer = use_padding_free_transformer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_dtensors_enabled():
            input = tensor_to_dtensor(input, current_placement=Shard(-1))
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=self.output_placement)
        else:
            assert not self.use_padding_free_transformer
            assert not self.sequence_parallel

            input = reduce_from_tensor_parallel_region(input, op=ReduceOp.SUM)
            input = F.linear(input, self.weight.to_local(), None if self.bias is None else self.bias.to_local())

        return input

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.bias is not None
        )
