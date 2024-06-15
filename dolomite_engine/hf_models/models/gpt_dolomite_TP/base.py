from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm

from ....utils import ProcessGroupManager, SafeTensorsWeightsManager
from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import ParameterizedEmbedding, RoPE, YaRNScaledRoPE, get_normalization_function
from ...modeling_utils_TP import Alibi_TP, Dropout_TP, Embedding_TP
from ..gpt_dolomite import GPTDolomiteConfig, GPTDolomiteModel, GPTDolomitePreTrainedModel
from .layer import GPTDolomiteBlock_TP


class GPTDolomitePreTrainedModel_TP(GPTDolomitePreTrainedModel):
    _no_split_modules = ["GPTDolomiteBlock_TP"]


class GPTDolomiteModel_TP(GPTDolomitePreTrainedModel_TP, GPTDolomiteModel):
    def __init__(self, config: GPTDolomiteConfig, tensor_parallel_embeddings: bool = False, **kwargs) -> None:
        GPTDolomitePreTrainedModel.__init__(self, config, **kwargs)

        self.tensor_parallel_embeddings = tensor_parallel_embeddings

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.head_dim = self.embed_dim // self.num_heads

        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        if self.tensor_parallel_embeddings:
            self.wte = Embedding_TP(config.vocab_size, self.embed_dim, std=self.initializer_range)
        else:
            self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else Dropout_TP(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                GPTDolomiteBlock_TP(
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

    def get_input_embeddings(self) -> Union[Embedding_TP, ParameterizedEmbedding]:
        return self.wte

    def set_input_embeddings(self, new_embeddings: Union[Embedding_TP, ParameterizedEmbedding]) -> None:
        self.wte = new_embeddings

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        # word embeddings
        if self.tensor_parallel_embeddings:
            self.wte.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wte.")
        else:
            state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "wte.weight")}
            self.wte.load_state_dict(state_dict)

        # positional embeddings
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            if self.tensor_parallel_embeddings:
                self.wpe.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wpe.")
            else:
                state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "wpe.weight")}
                self.wpe.load_state_dict(state_dict)
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            with torch.device(torch.cuda.current_device()):
                self.alibi.reset_parameters()
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            with torch.device(torch.cuda.current_device()):
                self.rope.reset_parameters()
        else:
            raise ValueError(f"unexpected position_embedding_type ({self.position_embedding_type})")

        # layers
        for layer_idx, block in tqdm(enumerate(self.h), desc="Loading layers"):
            block.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + f"h.{layer_idx}.")

        # final layernorm
        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_f.weight")}
        if hasattr(self.ln_f, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_f.bias")
        self.ln_f.load_state_dict(state_dict)

    def _get_alibi_bias(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.position_embedding_type != PositionEmbeddingType.alibi:
            return None

        alibi_bias = self.alibi(attention_mask, batch_size, key_length, device, dtype)

        if self._use_eager_attention:
            if self.attention_head_type == AttentionHeadType.mqa:
                if query_length != 1:
                    alibi_bias = alibi_bias.repeat(1, query_length, 1)
            elif self.attention_head_type in [AttentionHeadType.mha, AttentionHeadType.gqa]:
                alibi_bias = alibi_bias.unsqueeze(2)
                if query_length != 1:
                    alibi_bias = alibi_bias.expand(-1, -1, query_length, -1)
                alibi_bias = alibi_bias.view(
                    batch_size * (self.num_heads // self.tp_world_size), query_length, key_length
                )
            else:
                raise NotImplementedError()
        elif self._use_sdpa:
            alibi_bias = alibi_bias.unsqueeze(2)
            if query_length != 1:
                alibi_bias = alibi_bias.expand(-1, -1, query_length, -1)
        elif self._use_flash_attention_2:
            raise ValueError()

        return alibi_bias

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            if self.tensor_parallel_embeddings:
                self.wpe = Embedding_TP(max_position_embeddings, self.embed_dim)
            else:
                self.wpe = ParameterizedEmbedding(max_position_embeddings, self.embed_dim)
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            self.alibi = Alibi_TP(self.num_heads)
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.head_dim, max_position_embeddings=max_position_embeddings, base=self.config.rope_theta
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        else:
            raise NotImplementedError()
