import math

import torch
import torch.nn as nn

from .....utils import is_cute_kernels_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ..config import MoEDolomiteConfig
from .base import MoE, ParameterizedExperts


if is_cute_kernels_available():
    from cute_kernels.kernels import continuous_count_cute
    from cute_kernels.kernels.scattermoe.triton_implementation import scattered_experts


class ParameterizedScatteredExperts(ParameterizedExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        assert not add_bias, "scattermoe doesn't support bias"

        super().__init__(
            num_experts, in_features, out_features, add_bias=add_bias, device=device, dtype=dtype, std=std
        )

    def forward(
        self,
        input: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        return scattered_experts(
            inputs=input,
            expert_weights=self.weight.permute(0, 2, 1),
            k=k,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )


class ScatterMoE(MoE):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        nn.Module.__init__(self)

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.n_inner
        self.shared_intermediate_size = config.shared_n_inner

        activation_function = config.activation_function

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)
        residual_dropout = config.resid_pdrop

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=config.num_experts,
            bias=False,
            std=std,
        )

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = ParameterizedScatteredExperts(
            num_experts=config.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=config.add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_fc_shared = ParameterizedLinear(
                in_features=self.hidden_size,
                out_features=(
                    2 * self.shared_intermediate_size if is_glu(activation_function) else self.shared_intermediate_size
                ),
                bias=config.add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedScatteredExperts(
            num_experts=config.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=config.add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_proj_shared = ParameterizedLinear(
                in_features=self.shared_intermediate_size,
                out_features=self.hidden_size,
                bias=config.add_bias,
                std=std,
            )

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()

            if sorted_expert_idxs.is_cuda and is_cute_kernels_available():
                expert_offsets = continuous_count_cute(x=sorted_expert_idxs, size=self.num_experts).cumsum(-1)
            else:
                expert_offsets = sorted_expert_idxs.bincount(minlength=self.num_experts).cumsum(-1)

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_out=True,
        )
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            gates=router_weights,
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states
