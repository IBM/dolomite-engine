import torch

from .....utils import is_cute_kernels_available
from .base import MoE, ParameterizedExperts


if is_cute_kernels_available():
    from cute_kernels.kernels import continuous_count_cute
    from cute_kernels.kernels.scattermoe.triton_implementation import bincount, scattered_experts


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
    linear_class = ParameterizedScatteredExperts

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()

            if sorted_expert_idxs.is_cuda and is_cute_kernels_available() and self.is_hopper_or_newer_gpu:
                expert_offsets = continuous_count_cute(x=sorted_expert_idxs, size=self.num_experts).cumsum(-1)
            else:
                expert_offsets = bincount(sorted_expert_idxs, minlength=self.num_experts).cumsum(-1)

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
