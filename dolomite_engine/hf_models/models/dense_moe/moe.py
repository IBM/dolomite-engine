import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gpt_megatron.mlp import MLP
from .inference import mask_probability


class DenseMoE(MLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        activation_function: str,
        add_bias: bool = True,
        residual_dropout: float = 0.1,
        inference_method: dict = None,
    ) -> None:
        super().__init__(
            hidden_size,
            num_experts * intermediate_size,
            activation_function,
            add_bias,
            residual_dropout,
        )

        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.inference_method = inference_method

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ==========================================================================================
        # routing_weights -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        routing_weights = mask_probability(routing_weights, self.inference_method)

        # ==========================================================================================
        # routing_weights -> (batch_size, query_length, num_experts)
        # ==========================================================================================

        routing_weights = routing_weights.repeat_interleave(self.intermediate_size, dim=-1)

        # ==========================================================================================
        # routing_weights -> (batch_size, query_length, intermediate_size * num_experts)
        # ==========================================================================================

        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * routing_weights
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # ==========================================================================================
        # routing_weights -> (batch_size, query_length, num_heads * hidden_size)
        # ==========================================================================================

        return hidden_states
