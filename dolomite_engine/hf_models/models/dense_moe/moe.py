from copy import deepcopy

import torch
import torch.nn.functional as F

from ...modeling_utils import ParameterizedLinear
from ..gpt_dolomite.mlp import MLP
from .config import DenseMoEConfig
from .inference import mask_probability


class DenseMoE(MLP):
    def __init__(self, config: DenseMoEConfig, inference_method: dict | None = None) -> None:
        hidden_size = config.n_embd

        self.num_experts = config.num_experts
        self.intermediate_size = config.n_inner
        self.inference_method = inference_method

        config_copy = deepcopy(config)
        config_copy.n_inner = self.num_experts * self.intermediate_size
        super().__init__(config_copy)
        del config_copy

        self.gate = ParameterizedLinear(hidden_size, self.num_experts, bias=False)

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
