import torch
from transformers import Cache


class SoftmaxAttentionCache:
    def __init__(self) -> None:
        self._seen_tokens = 0
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None

    def get_cache(self) -> tuple[torch.Tensor | None]:
        return self.key_cache, self.value_cache

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor | None = None, sequence_length_dimension: int = -2
    ) -> tuple[torch.Tensor]:
        self._seen_tokens += key_states.size(sequence_length_dimension)

        if self.key_cache is None:
            self.key_cache = key_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=sequence_length_dimension)

        if value_states is not None:
            if self.value_cache is None:
                self.value_cache = value_states
            else:
                self.value_cache = torch.cat([self.value_cache, value_states], dim=sequence_length_dimension)

        return self.key_cache, self.value_cache

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> None:
        return None

    def crop(self, max_length: int):
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]


class DynamicCache(Cache):
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> list[tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __len__(self):
        return len(self.key_cache)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
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

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> None:
        return None

    def crop(self, max_length: int):
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> list["DynamicCache"]:
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: list["DynamicCache"]) -> "DynamicCache":
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx].numel()]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx].numel()]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
