from typing import Any

import torch
from transformers import Cache, DynamicCache


class _LinearCache(Cache):
    def __init__(self) -> None:
        self.states = []
        self._seen_tokens = 0

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
                self._seen_tokens += 1

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


class RNNCache(Cache):
    def __init__(self, attention_pattern: str) -> None:
        super().__init__()

        self.attention_pattern = attention_pattern

        self.cache_map = []
        num_attention_layers = 0
        num_linear_layers = 0
        for attention_map in attention_pattern:
            if attention_map == "a":
                self.cache_map.append((attention_map, num_attention_layers))
                num_attention_layers += 1
            elif attention_map == "d":
                self.cache_map.append((attention_map, num_linear_layers))
                num_linear_layers += 1
            else:
                raise ValueError("unexpected attention_pattern")

        self.attention_cache = DynamicCache()
        self.linear_cache = _LinearCache()

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        attention_map, layer_idx = self.cache_map[layer_idx]
        if attention_map == "a":
            state = self.attention_cache[layer_idx]
        elif attention_map == "d":
            state = self.linear_cache[layer_idx]
        else:
            raise ValueError()

        return state

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self[layer_idx]

    def __len__(self):
        return len(self.attention_cache) + len(self.linear_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor]:
        attention_map, layer_idx = self.cache_map[layer_idx]
        if attention_map == "a":
            key_states, value_states = self.attention_cache.update(
                key_states=key_states, value_states=value_states, layer_idx=layer_idx
            )
        elif attention_map == "d":
            assert value_states is None
            key_states = self.linear_cache.update(state=key_states, layer_idx=layer_idx)
        else:
            raise ValueError()

        return key_states, value_states

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if len(self) <= layer_idx:
            return 0
        return len(self.attention_cache)

    def get_max_length(self) -> int | None:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self.attention_cache.reorder_cache(beam_idx)
        self.linear_cache.reorder_cache(beam_idx)
