import torch
from transformers import DynamicCache
from transformers.models.jamba.modeling_jamba import (
    HybridMambaAttentionDynamicCache as _HybridMambaAttentionDynamicCache,
)

from ....utils import divide_if_divisible
from .config import Mamba2DolomiteConfig


class HybridMambaAttentionDynamicCache(_HybridMambaAttentionDynamicCache):
    def __init__(
        self,
        config: Mamba2DolomiteConfig,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
    ) -> None:
        DynamicCache.__init__(self)

        self.layers_block_type = config.layer_map
        self.has_previous_state = False  # only used by mamba
        ssm_state_size = config.ssm_state_size

        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba2":
                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        (config.ssm_intermediate_size + 2 * config.n_groups * ssm_state_size),
                        config.conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size,
                        config.ssm_num_heads,
                        divide_if_divisible(config.ssm_intermediate_size, config.ssm_num_heads),
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
