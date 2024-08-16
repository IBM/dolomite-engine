import torch.nn as nn

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from ...modeling_utils import Alibi, ParameterizedEmbedding, RoPE, get_normalization_function
from ...utils import convert_padding_free_lists_to_tensors, divide_if_divisible
from .config import GPTDolomiteConfig
from .layer import GPTDolomiteBlock


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock
    _no_split_modules = ["GPTDolomiteBlock"]


class GPTDolomiteModel(GPTDolomitePreTrainedModel, BaseModelMixin):
    def __init__(self, config: GPTDolomiteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        self.head_dim = divide_if_divisible(
            self.embed_dim,
            self.num_heads,
            f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                self.layer_class(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
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
