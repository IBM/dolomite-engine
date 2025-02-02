import torch
import torch.nn as nn


class GatedRMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        hidden_states = hidden_states.to(input_dtype)

        if self.weight is not None:
            hidden_states = self.weight * hidden_states

        return hidden_states
