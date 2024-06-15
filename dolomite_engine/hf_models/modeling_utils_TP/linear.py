import torch
import torch.distributed
import torch.nn.functional as F

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager
from ..modeling_utils import ParameterizedLinear
from ..utils import divide_if_divisible
from .TP import (
    copy_to_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    tensor_parallel_split_safetensor_slice,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = copy_to_tensor_parallel_region(input)
        input = super().forward(input)
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # we can't call super().forward here since that will add bias to each TP rank
        # but for tensor parallel, we need to add it on only 1 TP rank
        input = F.linear(input, self.weight, None)
        input = reduce_from_tensor_parallel_region(input)
        if self.bias is not None:
            input = input + self.bias
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
