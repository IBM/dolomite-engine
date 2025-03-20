import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from .....utils import ProcessGroupManager, is_cute_kernels_available
from ....enums import InitMethod
from ....loss import add_aux_loss
from ...activations import get_activation_function, is_glu
from ...linear import ParameterizedLinear
from ..mlp import _get_std_for_linear


if is_cute_kernels_available():
    from cute_kernels.kernels import continuous_count_cute
    from cute_kernels.kernels.scattermoe.triton_implementation import bincount


class ParameterizedExperts(nn.Module):
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
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features, device=device, dtype=dtype))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features, device=device, dtype=dtype))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(self, input: torch.Tensor, num_experts_per_token: torch.Tensor) -> torch.Tensor:
        input = input.split(num_experts_per_token.tolist(), dim=0)
        input = [
            F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]
        input = torch.cat(input, dim=0)
        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()


class MoE(nn.Module):
    linear_class = ParameterizedExperts

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: InitMethod,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        self.c_fc = self.linear_class(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_fc_shared = ParameterizedLinear(
                in_features=self.hidden_size,
                out_features=(
                    2 * self.shared_intermediate_size if is_glu(activation_function) else self.shared_intermediate_size
                ),
                bias=add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std /= math.sqrt(2 * num_layers)

        self.c_proj = self.linear_class(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_proj_shared = ParameterizedLinear(
                in_features=self.shared_intermediate_size,
                out_features=self.hidden_size,
                bias=add_bias,
                std=std,
            )

        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        moe_output = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits, probs=torch.softmax(router_logits, dim=-1), topk_idxs=selected_experts
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return hidden_states

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        total_q = hidden_states.shape[0]

        batch_index, batch_gates, num_experts_per_token = self._compute_expert_assignment(
            router_weights, selected_experts
        )

        hidden_states = hidden_states[batch_index]

        hidden_states = self.c_fc(hidden_states, num_experts_per_token)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states, num_experts_per_token)

        hidden_states = hidden_states * batch_gates.unsqueeze(-1)  # [:, None]
        zeros = torch.zeros((total_q, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, batch_index, hidden_states)

        return hidden_states

    def _compute_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc_shared(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj_shared(hidden_states)
        return hidden_states

    def _compute_expert_assignment(
        self, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor]:
        selected_experts = selected_experts.flatten()

        if selected_experts.is_cuda and is_cute_kernels_available():
            num_experts_per_token = continuous_count_cute(x=selected_experts, size=self.num_experts)
        else:
            num_experts_per_token = selected_experts.bincount(minlength=self.num_experts)

        # sort and group input tokens according to expert assignment
        _, index_sorted_experts = selected_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts // self.top_k  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        router_weights = router_weights.flatten()  # [num_tokens * top_k]
        batch_gates = router_weights[index_sorted_experts]  # [num_tokens * top_k]

        return batch_index, batch_gates, num_experts_per_token

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices

    def _compute_switch_loss(self, logits: torch.Tensor, probs: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        # Get target device from input tensors
        device = logits.device
        
        # Ensure all inputs are on the correct device
        logits = logits.to(device).view(-1, logits.size(-1))
        probs = probs.to(device).view(-1, probs.size(-1))
        topk_idxs = topk_idxs.to(device)

        num_experts = logits.size(1)
        acc_probs = probs.sum(0).to(device)

        # Ensure freq is explicitly created on the correct device
        if topk_idxs.is_cuda and is_cute_kernels_available() and self.is_hopper_or_newer_gpu:
            # Get freq and immediately move to correct device with correct dtype
            freq = continuous_count_cute(x=topk_idxs.flatten(), size=num_experts)
            freq = freq.to(device=device, dtype=logits.dtype)
        else:
            # Get freq and immediately move to correct device with correct dtype
            freq = bincount(topk_idxs.flatten(), minlength=num_experts)
            freq = freq.to(device=device, dtype=logits.dtype)

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            freq = all_reduce(freq, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())
            # Ensure it's on the right device after distributed operation
            freq = freq.to(device)

        # Now both tensors should definitely be on the same device
        normalized_acc_probs = F.normalize(acc_probs, p=1, dim=0)
        
        # Remove the redundant to(device) since freq is already on the correct device
        normalized_freq = F.normalize(freq, p=1, dim=0)

        switch_loss = num_experts * (normalized_acc_probs * normalized_freq).sum()
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss

        return loss
