from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite.mlp import MLP
from .config import MoEDolomiteConfig


class SparseMoE(nn.Module):
    def __init__(self, config: MoEDolomiteConfig, use_padding_free_transformer: bool) -> None:
        super().__init__()

        hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights

        # router
        self.gate = ParameterizedLinear(hidden_size, self.num_experts, bias=False)

        config_copy = deepcopy(config)
        config_copy.add_bias = False
        self.experts = nn.ModuleList([MLP(config_copy) for _ in range(self.num_experts)])
        del config_copy

        # shared bias amoung experts (Megablocks has shared bias for some reason)
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if config.add_bias else None

        self._use_padding_free_transformer = use_padding_free_transformer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_padding_free_transformer:
            _, hidden_dim = hidden_states.shape
        else:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits -> (batch_size * sequence_length, num_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.top_k == 1:
            routing_weights, selected_experts = routing_weights.max(dim=-1, keepdim=True)
        else:
            routing_weights, selected_experts = routing_weights.topk(self.top_k, dim=-1)

        if self.normalize_expert_weights:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask -> (num_experts, top_k, batch_size * sequence_length)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if not self._use_padding_free_transformer:
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.bias is not None:
            final_hidden_states += self.bias

        return final_hidden_states, router_logits

    def extra_repr(self) -> str | None:
        if self.bias is not None:
            return f"(bias): Parameter(size={tuple(self.bias.size())})"
