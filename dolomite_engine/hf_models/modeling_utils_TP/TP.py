from typing import Tuple

import torch
import torch.distributed

from ..modeling_utils import ParameterizedLinear
from ..parallel import ProcessGroupManager
from ..safetensors import SafeTensorsWeightsManager
from .random import CUDA_RNGStatesTracker


_TENSOR_PARALLEL_GROUP_MANAGER: ProcessGroupManager = None
_RNG_TRACKER: CUDA_RNGStatesTracker = None


def get_cuda_rng_tracker() -> CUDA_RNGStatesTracker:
    return _RNG_TRACKER


def set_cuda_rng_tracker(tracker: CUDA_RNGStatesTracker) -> None:
    global _RNG_TRACKER
    _RNG_TRACKER = tracker


def get_tensor_parallel_group_manager() -> ProcessGroupManager:
    return _TENSOR_PARALLEL_GROUP_MANAGER


def set_tensor_parallel_group_manager(process_group_manager: ProcessGroupManager) -> None:
    global _TENSOR_PARALLEL_GROUP_MANAGER
    _TENSOR_PARALLEL_GROUP_MANAGER = process_group_manager


def _reduce(input: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_parallel_group_manager().get_world_size() == 1:
        return input

    # All-reduce.
    torch.distributed.all_reduce(input, group=get_tensor_parallel_group_manager().get_process_group())

    return input


class CopyToTensorParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _reduce(grad_output)


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input: torch.Tensor) -> torch.Tensor:
        return _reduce(input)

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class ColumnParallelLinear(ParameterizedLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        assert (
            out_features % self.tp_world_size == 0
        ), f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})"

        self.out_features_per_device = out_features // self.tp_world_size
        super().__init__(in_features, self.out_features_per_device, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = CopyToTensorParallelRegion.apply(input)
        return super().forward(input)

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
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


class RowParallelLinear(ParameterizedLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()
        self.is_tp_first_rank = get_tensor_parallel_group_manager().get_first_rank() == torch.distributed.get_rank()

        assert (
            in_features % self.tp_world_size == 0
        ), f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})"

        self.in_features_per_device = in_features // self.tp_world_size
        self.out_features = out_features
        self.tp_bias = bias

        super().__init__(self.in_features_per_device, out_features, bias=bias if self.is_tp_first_rank else False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = super().forward(input)
        return ReduceFromTensorParallelRegion.apply(input)

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=1)
        state_dict = {"weight": weight}

        if self.bias is not None:
            assert (
                self.is_tp_first_rank
            ), "bias parameter found on rank that is not the first rank for the process group"

            bias = safetensors_weight_manager.get_tensor(prefix + "bias")
            state_dict["bias"] = bias

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.tp_bias
        )


def tensor_parallel_split_safetensor_slice(slice, dim: int, start_end: Tuple[int, int] = None) -> torch.Tensor:
    shape = slice.get_shape()
    dimensionality = len(shape)
    assert 1 <= dimensionality <= 2, f"tensor should be either 1 or 2 dimensional but {dimensionality} was found"

    tp_rank = get_tensor_parallel_group_manager().get_rank()
    tp_world_size = get_tensor_parallel_group_manager().get_world_size()

    if start_end is None:
        assert (
            shape[dim] % tp_world_size == 0
        ), f"split dimension ({dim}) is not divisible by tensor parallel world size ({tp_world_size})"
        stride = shape[dim] // tp_world_size
        start_index = tp_rank * stride
        end_index = (tp_rank + 1) * stride
    else:
        start_index = start_end[0]
        end_index = start_end[1]

    if dimensionality == 1:
        # bias tensor
        assert dim == 0, f"dim has to 0 for a bias tensor but dim ({dim}) was passed"
        return slice[start_index:end_index]
    elif dimensionality == 2:
        assert 0 <= dim <= 1, f"dim has to 0 or 1 for a weight tensor but dim ({dim}) was passed"
        # weight tensor
        if dim == 0:
            return slice[start_index:end_index, :]
        else:
            return slice[:, start_index:end_index]
    else:
        raise RuntimeError("this code should not be reachable")
