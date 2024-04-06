import torch.nn as nn

from ...enums import PositionEmbeddingType
from ...modeling_utils import get_normalization_function
from ..gpt_megatron import GPTMegatronPreTrainedModel
from ..moe_megablocks import MoEMegablocksModel, MoEMegablocksPreTrainedModel
from .config import DenseMoEConfig
from .layer import DenseMoEBlock


class DenseMoEPreTrainedModel(MoEMegablocksPreTrainedModel):
    config_class = DenseMoEConfig
    _no_split_modules = ["DenseMoEBlock"]

    def __init__(self, config: DenseMoEConfig, *inputs, **kwargs):
        GPTMegatronPreTrainedModel.__init__(self, config, *inputs, **kwargs)

        assert self._use_sdpa, "only SDPA is supported for dense_moe"
        self.inference_method = kwargs.get("inference_method")


class DenseMoEModel(DenseMoEPreTrainedModel, MoEMegablocksModel):
    def __init__(self, config: DenseMoEConfig, **kwargs) -> None:
        DenseMoEPreTrainedModel.__init__(self, config, **kwargs)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.mask_value = None

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                DenseMoEBlock(
                    config, self.normalization_implementation, layer_idx=i, inference_method=self.inference_method
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

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
