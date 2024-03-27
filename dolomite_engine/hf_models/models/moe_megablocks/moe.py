import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gpt_megatron.mlp import MLP


class SparseMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation_function: str,
        normalize_expert_weights: bool,
        add_bias: bool = True,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_expert_weights = normalize_expert_weights

        # router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                MLP(
                    hidden_size,
                    intermediate_size,
                    activation_function,
                    add_bias=False,
                    residual_dropout=residual_dropout,
                )
                for _ in range(self.num_experts)
            ]
        )

        # shared bias amoung experts (Megablocks has shared bias for some reason)
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if add_bias else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.bias is not None:
            final_hidden_states += self.bias

        return final_hidden_states, router_logits

    def extra_repr(self):
        if self.bias is not None:
            return f"(bias): Parameter(size={tuple(self.bias.size())})"
