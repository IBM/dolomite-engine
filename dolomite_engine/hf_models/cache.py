import torch
from transformers import DynamicCache
from transformers.models.jamba.modeling_jamba import (
    HybridMambaAttentionDynamicCache as _HybridMambaAttentionDynamicCache,
)

from ..utils import divide_if_divisible
from .config import CommonConfig


class HybridMambaAttentionDynamicCache(_HybridMambaAttentionDynamicCache):
    def __init__(
        self,
        config: CommonConfig,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
    ) -> None:
        DynamicCache.__init__(self)

        self.has_previous_state = False  # only used by mamba

        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_layers):
            block = config.sequence_mixer_blocks[i]

            if block.sequence_mixer_type == "mamba2":
                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        (block.intermediate_size + 2 * block.num_groups * block.state_size),
                        config.conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size,
                        block.num_heads,
                        divide_if_divisible(block.intermediate_size, block.num_heads, ""),
                        block.state_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_layers)]
