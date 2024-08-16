import torch.nn as nn

from ....utils import ProcessGroupManager
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ...modeling_utils_TP import Dropout_TP, Embedding_TP, get_normalization_function_TP
from ..gpt_dolomite import GPTDolomiteConfig
from .layer import GPTDolomiteBlock_TP


class GPTDolomitePreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock_TP
    _no_split_modules = ["GPTDolomiteBlock_TP"]

    def __init__(self, config: GPTDolomiteConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.tensor_parallel_word_embeddings = kwargs.get("tensor_parallel_word_embeddings", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, BaseModelMixin_TP):
    def __init__(self, config: GPTDolomiteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.head_dim = self.embed_dim // self.num_heads

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.wte = Embedding_TP(
            config.vocab_size,
            self.embed_dim,
            std=self.initializer_range,
            tensor_parallel_word_embeddings=self.tensor_parallel_word_embeddings,
            use_padding_free_transformer=self._use_padding_free_transformer,
            sequence_parallel=self.sequence_parallel,
        )

        self.drop = (
            nn.Identity()
            if config.embd_pdrop == 0
            else Dropout_TP(
                config.embd_pdrop,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )
        )
        self.h = nn.ModuleList(
            [
                self.layer_class(
                    config,
                    normalization_implementation=self.normalization_implementation,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    layer_idx=i,
                    sequence_parallel=self.sequence_parallel,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = get_normalization_function_TP(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
            use_padding_free_transformer=self._use_padding_free_transformer,
            sequence_parallel=self.sequence_parallel,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()
