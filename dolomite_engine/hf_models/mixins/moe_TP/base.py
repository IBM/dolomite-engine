import torch.nn as nn

from ....utils import divide_if_divisible
from ...config import CommonConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils_TP import Dropout_TP, Embedding_TP, get_normalization_function_TP
from ..dense_TP import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..moe import BaseMoEModelMixin, PreTrainedMoEModelMixin


class PreTrainedMoEModelMixin_TP(PreTrainedMoEModelMixin, PreTrainedModelMixin_TP):
    def __init__(self, config: CommonConfig, *args, **kwargs):
        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        super().__init__(config, *args, **kwargs)


class BaseMoEModelMixin_TP(BaseMoEModelMixin, BaseModelMixin_TP):
    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.head_dim = self.embed_dim // self.num_heads

        self.layers_per_stage = divide_if_divisible(
            config.n_layer, self.num_pipeline_stages, "layers should be divisible by num_pipeline_stages"
        )

        self.layer_start_id = self.layers_per_stage * self.pipeline_stage_id
        self.layer_end_id = self.layers_per_stage * (self.pipeline_stage_id + 1)

        if self.is_first_stage:
            self.wte = Embedding_TP(
                config.vocab_size,
                self.embed_dim,
                std=self.initializer_range,
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

        self.h = nn.ModuleDict(
            {
                str(i): self.layer_class(
                    config,
                    attention_implementation=self.attention_implementation,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    moe_implementation=self.moe_implementation,
                    layer_idx=i,
                    sequence_parallel=self.sequence_parallel,
                )
                for i in range(self.layer_start_id, self.layer_end_id)
            }
        )

        if self.is_last_stage:
            self.ln_f = get_normalization_function_TP(
                config.normalization_function,
                self.embed_dim,
                eps=config.layer_norm_epsilon,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()
