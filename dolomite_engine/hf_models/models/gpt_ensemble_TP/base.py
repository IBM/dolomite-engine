import torch.nn as nn

from ....utils import ProcessGroupManager
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import get_normalization_function
from ...modeling_utils_TP import Dropout_TP, Embedding_TP
from ..gpt_dolomite_TP import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP
from ..gpt_ensemble import GPTEnsembleConfig
from .layer import GPTEnsembleBlock_TP


class GPTEnsemblePreTrainedModel_TP(GPTDolomitePreTrainedModel_TP):
    config_class = GPTEnsembleConfig
    _no_split_modules = ["GPTEnsembleBlock_TP"]


class GPTEnsembleModel_TP(GPTEnsemblePreTrainedModel_TP, GPTDolomiteModel_TP):
    def __init__(self, config: GPTEnsembleConfig, tensor_parallel_embeddings: bool = False, **kwargs) -> None:
        GPTEnsemblePreTrainedModel_TP.__init__(self, config, **kwargs)

        self.tensor_parallel_embeddings = tensor_parallel_embeddings

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
            tensor_parallel_embeddings=self.tensor_parallel_embeddings,
        )

        self.drop = nn.Identity() if config.embd_pdrop == 0 else Dropout_TP(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                GPTEnsembleBlock_TP(
                    config,
                    self.normalization_implementation,
                    self.attention_implementation,
                    self._use_padding_free_transformer,
                    layer_idx=i,
                )
                for i in range(config.num_hidden_layers)
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
