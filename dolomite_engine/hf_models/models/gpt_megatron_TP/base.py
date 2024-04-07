from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import RoPE, YaRNScaledRoPE, get_normalization_function
from ...modeling_utils_TP import Alibi_TP, Dropout_TP, Embedding_TP, get_tensor_parallel_group_manager
from ...safetensors import SafeTensorsWeightsManager
from ..gpt_megatron import GPTMegatronConfig, GPTMegatronModel, GPTMegatronPreTrainedModel
from .layer import GPTMegatronBlock_TP


class GPTMegatronModel_TP(GPTMegatronModel):
    def __init__(
        self,
        config: GPTMegatronConfig,
        tensor_parallel_vocab_matrix: bool = False,
        tensor_parallel_position_embedding_matrix: bool = False,
        **kwargs,
    ) -> None:
        GPTMegatronPreTrainedModel.__init__(self, config, **kwargs)

        self.tensor_parallel_vocab_matrix = tensor_parallel_vocab_matrix
        self.tensor_parallel_position_embedding_matrix = tensor_parallel_position_embedding_matrix

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = self.embed_dim // self.num_heads

        self.tp_world_size = get_tensor_parallel_group_manager().get_world_size()

        if self.tensor_parallel_vocab_matrix:
            self.wte = Embedding_TP(config.vocab_size, self.embed_dim)
        else:
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.drop = Dropout_TP(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                GPTMegatronBlock_TP(
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

    def get_input_embeddings(self) -> Union[Embedding_TP, nn.Embedding]:
        return self.wte

    def set_input_embeddings(self, new_embeddings: Union[Embedding_TP, nn.Embedding]) -> None:
        self.wte = new_embeddings

    def load_unsharded_weights(self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = "") -> None:
        self._load_embeddings(self.wte, safetensors_weight_manager, prefix + "wte.")
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self._load_embeddings(self.wpe, safetensors_weight_manager, prefix + "wpe.")

        for layer_idx, block in tqdm(enumerate(self.h), desc="Loading layers"):
            block.load_unsharded_weights(safetensors_weight_manager, prefix + f"h.{layer_idx}.")

        state_dict = {
            "weight": safetensors_weight_manager.get_tensor(prefix + "ln_f.weight"),
            "bias": safetensors_weight_manager.get_tensor(prefix + "ln_f.bias"),
        }
        self.ln_f.load_state_dict(state_dict)

    def _load_embeddings(
        self,
        module: Union[Embedding_TP, nn.Embedding],
        safetensors_weight_manager: SafeTensorsWeightsManager,
        prefix: str,
    ) -> None:
        if isinstance(module, Embedding_TP):
            module.load_unsharded_weights(safetensors_weight_manager, prefix)
        else:
            state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "weight")}
            module.load_state_dict(state_dict)

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

        # ==========================================================================================
        # alibi_bias -> (batch_size, num_heads, key_length)
        # ==========================================================================================

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

        # ==========================================================================================
        # eager:
        #     AttentionHeadType.mqa:
        #         alibi_bias -> (batch_size, query_length * num_heads, key_length)
        #     AttentionHeadType.mha:
        #         alibi_bias -> (batch_size * num_heads, query_length, key_length)
        # sdpa:
        #     alibi_bias -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        return alibi_bias

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            if self.tensor_parallel_position_embedding_matrix:
                self.wpe = Embedding_TP(max_position_embeddings, self.embed_dim)
            else:
                self.wpe = nn.Embedding(max_position_embeddings, self.embed_dim)
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
