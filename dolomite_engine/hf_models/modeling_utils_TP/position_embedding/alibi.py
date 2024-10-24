import math

import torch

from ....utils import ProcessGroupManager, divide_if_divisible
from ...modeling_utils import Alibi


class Alibi_TP(Alibi):
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

        tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        num_heads_tp = divide_if_divisible(self.num_heads, ProcessGroupManager.get_tensor_parallel_world_size(), "")
        slopes = slopes[tp_rank * num_heads_tp : (tp_rank + 1) * num_heads_tp]

        self.register_buffer("slopes", slopes, persistent=False)
