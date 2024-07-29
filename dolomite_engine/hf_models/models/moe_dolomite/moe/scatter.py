import math

import torch
import torch.nn as nn

from .....utils import is_scattermoe_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ..config import MoEDolomiteConfig
from .base import SparseMoE


if is_scattermoe_available():
    import scattermoe
    from scattermoe.parallel_experts import parallel_linear as scattered_experts


class ScatterMoE(SparseMoE):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.gate = ParameterizedLinear(self.hidden_size, self.num_experts, bias=False)

        intermediate_size = config.n_inner
        activation_function = config.activation_function
        residual_dropout = config.resid_pdrop

        assert not config.add_bias, "ScatterMoE does not support add_bias"

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ParameterizedScatteredExperts(
            self.num_experts,
            self.hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedScatteredExperts(self.num_experts, intermediate_size, self.hidden_size, std=std)

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def _compute_expert_outputs(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

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
            gates=routing_weights,
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ParameterizedScatteredExperts(ParameterizedLinear):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        nn.Module.__init__(self)

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.std = std

        self.weight = nn.Parameter(torch.empty((num_experts, out_features, in_features), device=device, dtype=dtype))
        self.bias = None

        self.reset_parameters()

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
        results = scattered_experts(
            inputs,
            self.weight.permute(0, 2, 1),
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

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )
