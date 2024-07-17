import torch
import torch.nn as nn

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from ...utils import divide_if_divisible
from ..gpt_dolomite import GPTDolomiteModel, GPTDolomitePreTrainedModel
from .config import GPTEnsembleConfig
from .layer import GPTEnsembleBlock


class GPTEnsemblePreTrainedModel(GPTDolomitePreTrainedModel):
    config_class = GPTEnsembleConfig
    _no_split_modules = ["GPTEnsembleBlock"]


class GPTEnsembleModel(GPTEnsemblePreTrainedModel, GPTDolomiteModel):
    def __init__(self, config: GPTEnsembleConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.tensor_parallel_size = config.pretraining_tensor_parallel_size

        self.pretraining_tp_size = config.pretraining_tensor_parallel_size
        self.embed_dim_per_tp_rank = divide_if_divisible(self.embed_dim, self.pretraining_tp_size, "")

        self.head_dim = divide_if_divisible(
            self.embed_dim,
            self.num_heads,
            f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                GPTEnsembleBlock(
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
        if self.position_embedding_type == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently GPTEnsemble doesn't support alibi")

        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        rope_cos_sin = super()._get_rope_cos_sin(key_length, position_ids, dtype=dtype, device=device)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            cos, sin = rope_cos_sin

            if position_ids.shape[0] != 1:
                cos = cos.repeat(self.tensor_parallel_size, 1, 1, 1)
                sin = sin.repeat(self.tensor_parallel_size, 1, 1, 1)

            return cos, sin
