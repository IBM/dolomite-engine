import torch
import torch.nn.functional as F

from ....utils import SafeTensorsWeightsManager
from ...modeling_utils_TP import RowParallelLinear, tensor_parallel_split_safetensor_slice
from .TP import ensemble_reduce_from_tensor_parallel_region


class EnsembleRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.linear(input, self.weight, self.bias)
        input = ensemble_reduce_from_tensor_parallel_region(input)
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
