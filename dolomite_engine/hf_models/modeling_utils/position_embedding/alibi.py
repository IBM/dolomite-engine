import math

import torch
import torch.nn as nn


class Alibi(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.reset_parameters()

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if attention_mask is None:
            arange_tensor = (
                torch.arange(key_length, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            arange_tensor = (attention_mask.cumsum(dim=-1) - 1).masked_fill_(attention_mask == 0, 0).unsqueeze(1)

        alibi = self.slopes.unsqueeze(1) * arange_tensor
        return alibi.to(dtype)

    def reset_parameters(self) -> None:
        closest_power_of_2 = 2 ** math.floor(math.log2(self.num_heads))
        base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != self.num_heads:
            extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, self.num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        self.register_buffer("slopes", slopes, persistent=False)
