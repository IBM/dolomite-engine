from typing import Any, Mapping

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.placement_types import _Partial as Partial
from torch.profiler import record_function

from ...utils import (
    ProcessGroupManager,
    SafeTensorsWeightsManager,
    get_cuda_rng_tracker,
    is_dtensors_computation_enabled,
)
from ..modeling_utils import ParameterizedLinear
from ..utils import divide_if_divisible
from .TP import (
    copy_to_tensor_parallel_region,
    dtensor_to_tensor,
    modify_state_dict_to_dtensor_dict,
    reduce_from_tensor_parallel_region,
    tensor_parallel_split_safetensor_slice,
    tensor_to_dtensor,
)


class ColumnParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
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

        # TODO activate this hook if we drop the non-dtensor path, until then use functions
        # self.register_forward_pre_hook(partial(tensor_to_dtensor_hook, current_placement=Replicate()))
        # self.register_forward_hook(partial(dtensor_to_tensor_hook, desired_placement=Shard(-1)))

    @record_function("TP:column_parallel_linear")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_dtensors_computation_enabled():
            input = tensor_to_dtensor(input, current_placement=Replicate())
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=Shard(-1))
        else:
            input = copy_to_tensor_parallel_region(input)
            input = F.linear(
                input, weight=self.weight.to_local(), bias=None if self.bias is None else self.bias.to_local()
            )

        return input

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
        state_dict = {"weight": weight}

        if self.bias is not None:
            bias = safetensors_weight_manager.get_slice(prefix + "bias")
            bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
            state_dict["bias"] = bias

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class RowParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

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

        # TODO activate this hook if we drop the non-dtensor path, until then use functions
        # self.register_forward_pre_hook(partial(tensor_to_dtensor_hook, current_placement=Shard(-1)))
        # self.register_forward_hook(partial(dtensor_to_tensor_hook, desired_placement=Replicate()))

    @record_function("TP:row_parallel_linear")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_dtensors_computation_enabled():
            input = tensor_to_dtensor(input, current_placement=Shard(-1))
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=Replicate())
        else:
            input = F.linear(input, self.weight.to_local(), None)
            input = reduce_from_tensor_parallel_region(input)
            if self.bias is not None:
                input = input + self.bias.to_local()

        return input

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=1)
        state_dict = {"weight": weight}

        if self.bias is not None:
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "bias")

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.bias is not None
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class TensorParallelSharedLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
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

        # TODO activate this hook if we drop the non-dtensor path, until then use functions
        # self.register_forward_pre_hook(partial(tensor_to_dtensor_hook, current_placement=Replicate()))
        # self.register_forward_hook(
        #     partial(dtensor_to_tensor_hook, desired_placement=Replicate(), grad_placement=Partial())
        # )

    @record_function("TP:shared_linear")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if is_dtensors_computation_enabled():
            input = tensor_to_dtensor(input, current_placement=Replicate())
            input = super().forward(input)
            input = dtensor_to_tensor(input, desired_placement=Replicate(), grad_placement=Partial())
        else:
            input = F.linear(
                input, weight=self.weight.to_local(), bias=None if self.bias is None else self.bias.to_local()
            )
            input = copy_to_tensor_parallel_region(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        with get_cuda_rng_tracker().fork():
            return super().reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)
