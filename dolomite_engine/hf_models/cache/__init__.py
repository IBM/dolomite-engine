import torch
from transformers import Cache, DynamicCache
from transformers.models.jamba.modeling_jamba import (
    HybridMambaAttentionDynamicCache as _HybridMambaAttentionDynamicCache,
)

from ...utils import divide_if_divisible
from ..config import CommonConfig
from .attention import SoftmaxAttentionCache


class GenerationCache(Cache):
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.cache: list[SoftmaxAttentionCache] = []

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor]:
        if layer_idx < len(self):
            return self.cache[layer_idx].get_cache()

        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self.cache[layer_idx].get_cache()

    def __len__(self):
        return self.seen_tokens

    def update(self, layer_idx: int, **kwargs) -> tuple[torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif not self.key_cache[
            layer_idx
        ].numel():  # prefers not t.numel() to len(t) == 0 to export the model  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.cache[layer_idx].update(**kwargs)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.cache[layer_idx].get_seq_length()


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
                        block.conv_kernel_size,
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
