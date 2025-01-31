import torch
import torch.nn as nn


class GatedRMSNorm(nn.RMSNorm):
    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight is not None:
            hidden_states = self.weight * hidden_states.to(input_dtype)

        return hidden_states
