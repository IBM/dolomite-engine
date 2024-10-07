import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from .....utils import ProcessGroupManager, is_kernel_hyperdrive_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedTransposedLinear, get_activation_function, is_glu
from ....modeling_utils_TP import Dropout_TP, DTensorModule, dtensor_to_tensor, tensor_to_dtensor
from ....utils import divide_if_divisible
from ...moe_dolomite import MoEDolomiteConfig
from ...moe_dolomite.moe import ScatterMoE
from ...moe_dolomite.moe.scatter import ParameterizedScatteredExperts


if is_kernel_hyperdrive_available():
    from khd.kernels.scattermoe.triton_implementation import padded_block_indices, scattered_experts


class ReplicatedTransposedLinear_TP(ParameterizedTransposedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype, std=std
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )


class ParallelScatteredExperts(ParameterizedScatteredExperts, DTensorModule):
    expert_parallel_on_data_parallel_sharding_mesh: bool = False

    def forward(
        self,
        input: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        padded_block_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        return scattered_experts(
            input,
            self.weight.to_local().permute(1, 2, 0),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

    def _get_expert_parallel_world_size(self) -> int:
        return (
            ProcessGroupManager.get_data_parallel_sharding_world_size()
            if self.expert_parallel_on_data_parallel_sharding_mesh
            else ProcessGroupManager.get_data_parallel_world_size()
        )

    def _get_num_experts_per_expert_parallel_unit(self, num_experts: int) -> int:
        ep_world_size = self._get_expert_parallel_world_size()

        return divide_if_divisible(
            num_experts,
            ep_world_size,
            f"`num_experts` ({num_experts}) must be divisible by `expert_parallel_world_size` ({ep_world_size})",
        )


class ColumnParallelScatteredExperts(ParallelScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        expert_parallel: bool = False,
        expert_parallel_on_data_parallel_sharding_mesh: bool = False,
    ) -> None:
        self.expert_parallel_on_data_parallel_sharding_mesh = expert_parallel_on_data_parallel_sharding_mesh

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        out_features_per_device = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        num_experts_dim = 1
        out_features_dim = 0

        if expert_parallel:
            num_experts_per_device = self._get_num_experts_per_expert_parallel_unit(num_experts)

            super().__init__(
                num_experts=num_experts_per_device,
                in_features=in_features,
                out_features=out_features_per_device,
                add_bias=add_bias,
                device=device,
                dtype=dtype,
                std=std,
            )

            self.weight = nn.Parameter(
                DTensor.from_local(
                    self.weight,
                    device_mesh=(
                        ProcessGroupManager.get_mesh()["fsdp", "tp"]
                        if expert_parallel_on_data_parallel_sharding_mesh
                        else ProcessGroupManager.get_mesh()["ddp", "fsdp", "tp"]
                    ),
                    placements=(
                        [Shard(num_experts_dim), Shard(out_features_dim)]
                        if expert_parallel_on_data_parallel_sharding_mesh
                        else [Shard(num_experts_dim), Shard(num_experts_dim), Shard(out_features_dim)]
                    ),
                    run_check=False,
                )
            )
        else:
            super().__init__(
                num_experts=num_experts,
                in_features=in_features,
                out_features=out_features_per_device,
                add_bias=add_bias,
                device=device,
                dtype=dtype,
                std=std,
            )

            self.weight = nn.Parameter(
                DTensor.from_local(
                    self.weight,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    placements=[Shard(out_features_dim)],
                    run_check=False,
                )
            )


class RowParallelScatteredExperts(ColumnParallelScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        expert_parallel: bool = False,
        expert_parallel_on_data_parallel_sharding_mesh: bool = False,
    ) -> None:
        self.expert_parallel_on_data_parallel_sharding_mesh = expert_parallel_on_data_parallel_sharding_mesh

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        in_features_per_device = divide_if_divisible(
            in_features,
            tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        num_experts_dim = 1
        in_features_dim = 2

        if expert_parallel:
            num_experts_per_device = self._get_num_experts_per_expert_parallel_unit(num_experts)

            super().__init__(
                num_experts=num_experts_per_device,
                in_features=in_features_per_device,
                out_features=out_features,
                add_bias=add_bias,
                device=device,
                dtype=dtype,
                std=std,
            )

            self.weight = nn.Parameter(
                DTensor.from_local(
                    self.weight,
                    device_mesh=(
                        ProcessGroupManager.get_mesh()["fsdp", "tp"]
                        if expert_parallel_on_data_parallel_sharding_mesh
                        else ProcessGroupManager.get_mesh()["ddp", "fsdp", "tp"]
                    ),
                    placements=(
                        [Shard(num_experts_dim), Shard(in_features_dim)]
                        if expert_parallel_on_data_parallel_sharding_mesh
                        else [Shard(num_experts_dim), Shard(num_experts_dim), Shard(in_features_dim)]
                    ),
                    run_check=False,
                )
            )
        else:
            super().__init__(
                num_experts=num_experts,
                in_features=in_features_per_device,
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
                    placements=[Shard(in_features_dim)],
                    run_check=False,
                )
            )


class ScatterMoE_TP(ScatterMoE, DTensorModule):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
        expert_parallel: bool = False,
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

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.mesh = ProcessGroupManager.get_mesh()

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.gate = ReplicatedTransposedLinear_TP(
            in_features=self.hidden_size,
            out_features=self.num_experts,
            bias=False,
            std=std,
        )

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ColumnParallelScatteredExperts(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=config.add_bias,
            std=std,
            expert_parallel=expert_parallel,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = RowParallelScatteredExperts(
            num_experts=self.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=config.add_bias,
            std=std,
            expert_parallel=expert_parallel,
        )

        self.dropout = nn.Identity() if residual_dropout == 0 else Dropout_TP(residual_dropout)

        self.placement = Shard(0) if sequence_parallel else Replicate()
        self.expert_parallel = expert_parallel

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = tensor_to_dtensor(hidden_states, device_mesh=self.tp_mesh, current_placement=self.placement)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        hidden_states = dtensor_to_tensor(
            hidden_states,
            device_mesh=self.mesh,
            desired_placement=[Shard(0), Shard(0), Replicate()],
            grad_placement=[Shard(0), Shard(0), Partial()],
        )

        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)

        hidden_states = tensor_to_dtensor(hidden_states, device_mesh=self.tp_mesh, current_placement=Partial())
        hidden_states = dtensor_to_tensor(
            hidden_states, device_mesh=self.tp_mesh, desired_placement=self.placement, grad_placement=self.placement
        )

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = self._compute_switch_loss(
            logits=router_logits, probs=torch.softmax(router_logits, dim=-1), topk_idxs=selected_experts
        )

        return hidden_states, router_logits, aux_loss

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        router_logits = dtensor_to_tensor(
            router_logits, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
        )
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            padded_block_idxs, expert_offsets = padded_block_indices(sorted_expert_idxs, self.num_experts)

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        )
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=router_weights,
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states
