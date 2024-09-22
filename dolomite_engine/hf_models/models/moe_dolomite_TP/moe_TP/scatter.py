import math

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from .....utils import ProcessGroupManager, is_kernel_hyperdrive_available
from ....enums import InitMethod
from ....modeling_utils import get_activation_function, is_glu
from ....modeling_utils_TP import (
    DTensorModule,
    ReplicatedLinear,
    dtensor_to_tensor,
    get_module_placements,
    tensor_to_dtensor,
)
from ....utils import divide_if_divisible
from ...moe_dolomite import MoEDolomiteConfig
from ...moe_dolomite.moe import ScatterMoE
from ...moe_dolomite.moe.scatter import ParameterizedScatteredExperts


if is_kernel_hyperdrive_available():
    from khd.scattermoe.triton_implementation import padded_block_indices, scattered_experts


class ColumnParallelScatteredExperts(ParameterizedScatteredExperts, DTensorModule):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
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
            num_experts=num_experts,
            in_features=in_features,
            out_features=self.out_features_per_device,
            add_bias=add_bias,
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

        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

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
        # F.linear manually triggers an all gather for sequence parallel but custom kernels are not aware of the placements
        # so we manually call an all gather here
        inputs = tensor_to_dtensor(inputs, current_placement=self.input_placement)
        inputs = dtensor_to_tensor(inputs, desired_placement=Replicate(), grad_placement=Partial())

        weight = self.weight.to_local()

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


class RowParallelScatteredExperts(ParameterizedScatteredExperts, DTensorModule):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
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
            add_bias=add_bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                placements=[Shard(-1)],
                run_check=False,
            )
        )

        self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

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
        weight = self.weight.to_local()

        inputs = scattered_experts(
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

        inputs = tensor_to_dtensor(inputs, current_placement=Partial())
        inputs = dtensor_to_tensor(inputs, desired_placement=self.output_placement)

        return inputs


class ScatterMoE_TP(ScatterMoE, DTensorModule):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.n_inner

        activation_function = config.activation_function

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)
        residual_dropout = config.resid_pdrop

        self.gate = ReplicatedLinear(
            in_features=self.hidden_size,
            out_features=config.num_experts,
            bias=False,
            std=config.initializer_range,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=False,
        )

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)

        self.c_fc = ColumnParallelScatteredExperts(
            num_experts=config.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=config.add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = RowParallelScatteredExperts(
            num_experts=config.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=config.add_bias,
            std=std,
        )

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)
