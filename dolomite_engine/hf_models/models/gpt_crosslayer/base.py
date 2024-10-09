from collections import defaultdict

import torch.nn as nn

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from .config import GPTCrossLayerConfig
from .layer import GPTCrossLayerBlock


class GPTCrossLayerPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTCrossLayerConfig
    layer_class = GPTCrossLayerBlock
    _no_split_modules = ["GPTCrossLayerBlock"]

    def __init__(self, config: GPTCrossLayerConfig, *args, **kwargs):
        self.sharing_pattern = config.sharing_pattern
        super().__init__(config, *args, **kwargs)


class GPTCrossLayerModel(GPTCrossLayerPreTrainedModel, BaseModelMixin):
    def _init_model(self, config: GPTCrossLayerConfig, **kwargs) -> None:
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)

        global_index, local_index = 0, 0
        # layer_index to (global_index, local_index)
        self.layer_map = [(global_index, local_index)]

        # global_index to layer_index
        sub_layer_map = defaultdict(list)
        sub_layer_map[global_index].append(0)

        for layer_idx in range(1, config.n_layer):
            if self.sharing_pattern[layer_idx] != self.sharing_pattern[layer_idx - 1]:
                global_index += 1
                local_index = 0
            else:
                local_index += 1

            self.layer_map.append((global_index, local_index))
            sub_layer_map[global_index].append(layer_idx)

        self.h = nn.ModuleList(
            [
                self.layer_class(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_indices=sub_layer_map[i],
                    layer_idx=i,
                )
                for i in sub_layer_map
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def get_global_local_idx(self, index: int) -> tuple[int, int]:
        return self.layer_map[index]
