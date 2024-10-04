import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from .....utils import ProcessGroupManager, is_kernel_hyperdrive_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
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
    from khd.kernels.scattermoe.triton_implementation import scattered_experts


class ReplicatedRouter(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
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


class ColumnParallelScatteredExperts(ParameterizedScatteredExperts):
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
                placements=[Shard(0)],
                run_check=False,
            )
        )
        # Put in MLP
        # self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

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

        weight = self.weight.to_local()

        results = scattered_experts(
            inputs,
            weight.permute(1, 2, 0),
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

        # self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

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
            weight.permute(1, 2, 0),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

        return inputs


class ScatterMoE_TP(ScatterMoE, DTensorModule):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
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

        self.gate = ReplicatedRouter(
            in_features=self.hidden_size,
            out_features=config.num_experts,
            bias=False,
            std=config.initializer_range,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=False,  # replicate even if SP
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
        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)
        self.output_placement = self.input_placement

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        router_logits = dtensor_to_tensor(router_logits, desired_placement=Replicate(), grad_placement=Partial())
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = tensor_to_dtensor(hidden_states, current_placement=self.input_placement)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        hidden_states = dtensor_to_tensor(hidden_states, desired_placement=Replicate(), grad_placement=Partial())

        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)

        hidden_states = tensor_to_dtensor(hidden_states, current_placement=Partial())
        hidden_states = dtensor_to_tensor(
            hidden_states, desired_placement=self.output_placement, grad_placement=self.output_placement
        )

        hidden_states = self.dropout(hidden_states)
        return hidden_states, router_logits  # TODO include the auxiliary loss output.
