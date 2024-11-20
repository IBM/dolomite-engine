from typing import Any

import torch
from fla.models.utils import Cache as RNNCache
from transformers import Cache, DynamicCache


class _LinearCache(Cache):
    def __init__(self, attention_pattern: str, seen_tokens: int = 0) -> None:
        self.states = []
        self._seen_tokens = seen_tokens

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        state: tuple[torch.Tensor],
        layer_idx: int,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor]:
        if isinstance(state, torch.Tensor):
            state = (state,)
        if len(self.states) <= layer_idx:
            self.states.append(state)
        else:
            for i, s in enumerate(state):
                self.states[layer_idx][i].copy_(s)
            # update the number of seen tokens once we achieve the last layer
            if layer_idx == len(self) - 1:
                self._seen_tokens += offset

        return state

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> int | None:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.states)):
            device = self.states[layer_idx].device
            self.states[layer_idx] = self.states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> tuple[torch.Tensor]:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[torch.Tensor] | None = None, seen_tokens: int = 0) -> Cache:
        cache = cls(seen_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                cache.update(past_key_values[layer_idx], layer_idx)
        return cache
