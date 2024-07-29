from typing import Any, Mapping

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.placement_types import _Partial as Partial

from .....utils import ProcessGroupManager, is_scattermoe_available
from ....modeling_utils import ParameterizedLinear
from ....modeling_utils_TP import dtensor_to_tensor, modify_state_dict_to_dtensor_dict, tensor_to_dtensor
from ....utils import divide_if_divisible
from ..moe.scatter import ParameterizedScatteredExperts


if is_scattermoe_available():
    from scattermoe.parallel_experts import parallel_linear as scattered_experts


class ReplicatedParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:

        super().__init__(
            in_features=in_features, out_features=out_features, device=device, dtype=dtype, std=std, bias=False
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )
        self.input_placement = Replicate()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=self.input_placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=Replicate())
        return input

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class ColumnParallelScatteredExperts(ParameterizedScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.out_features_per_device = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=self.out_features_per_device,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                placements=[Shard(1)],
                run_check=False,
            )
        )

        self.input_placement = Replicate()

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        weight = dtensor_to_tensor(self.weight, desired_placement=Shard(1))

        results = scattered_experts(
            inputs,
            weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

        return results

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class RowParallelScatteredExperts(ParameterizedScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.in_features_per_device = divide_if_divisible(
            in_features,
            tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            num_experts=num_experts,
            in_features=self.in_features_per_device,
            out_features=out_features,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                placements=[Shard(2)],
                run_check=False,
            )
        )

        self.input_placement = Shard(-1)

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        weight = dtensor_to_tensor(self.weight, desired_placement=Shard(2))

        results = scattered_experts(
            inputs,
            weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

        results = tensor_to_dtensor(results, current_placement=Partial())
        results = dtensor_to_tensor(results, desired_placement=Replicate())

        return results

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)
