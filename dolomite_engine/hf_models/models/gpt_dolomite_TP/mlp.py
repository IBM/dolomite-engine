import torch
import torch.nn as nn

from ....utils import SafeTensorsWeightsManager
from ...modeling_utils import get_activation_function, is_glu
from ...modeling_utils_TP import (
    ColumnParallelLinear,
    Dropout_TP,
    RowParallelLinear,
    get_tensor_parallel_group_manager,
    tensor_parallel_split_safetensor_slice,
)
from ..gpt_dolomite.mlp import MLP


class MLP_TP(MLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool = True,
        residual_dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)

        self.tp_rank = get_tensor_parallel_group_manager().get_rank()
        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        self.is_glu_activation = is_glu(activation_function)
        self.add_bias = add_bias

        self.c_fc = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size if self.is_glu_activation else intermediate_size,
            bias=add_bias,
        )
        self.act = get_activation_function(activation_function)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=add_bias)
        self.dropout = Dropout_TP(residual_dropout)

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        # GLU is a special case and needs to be handled explicitely
        if self.is_glu_activation:
            weight = safetensors_weight_manager.get_slice(prefix + "c_fc.weight")

            shape = weight.get_shape()
            assert (
                shape[0] % (self.tp_world_size * 2) == 0
            ), f"split dimension ({0}) is not divisible by 2 x tensor parallel world size (2 x {self.tp_world_size})"
            stride = shape[0] // (self.tp_world_size * 2)

            # split weight tensors into gate and non-gate
            start_end = (self.tp_rank * stride, (self.tp_rank + 1) * stride)
            weight_1 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
            if self.add_bias:
                bias = safetensors_weight_manager.get_slice(prefix + "c_fc.bias")
                bias_1 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

            start_end = (
                (self.tp_world_size + self.tp_rank) * stride,
                (self.tp_world_size + self.tp_rank + 1) * stride,
            )
            weight_2 = tensor_parallel_split_safetensor_slice(weight, 0, start_end)
            if self.add_bias:
                bias_2 = tensor_parallel_split_safetensor_slice(bias, 0, start_end)

            state_dict = {"weight": torch.cat([weight_1, weight_2])}
            if self.add_bias:
                state_dict["bias"] = torch.cat([bias_1, bias_2])

            self.c_fc.load_state_dict(state_dict)
        else:
            self.c_fc.load_unsharded_weights(safetensors_weight_manager, prefix=prefix + "c_fc.")

        self.c_proj.load_unsharded_weights(safetensors_weight_manager, prefix=prefix + "c_proj.")
