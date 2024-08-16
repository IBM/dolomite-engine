import torch
from tqdm import tqdm

from ....utils import SafeTensorsWeightsManager
from ...config import CommonConfig
from ...enums import PositionEmbeddingType
from ...modeling_utils import RoPE, YaRNScaledRoPE
from ...modeling_utils_TP import Alibi_TP, Embedding_TP
from ..dense import BaseModelMixin, PreTrainedModelMixin


class PreTrainedModelMixin_TP(PreTrainedModelMixin):
    def __init__(self, config: CommonConfig, *args, **kwargs):
        PreTrainedModelMixin.__init__(self, config, *args, **kwargs)

        self.tensor_parallel_word_embeddings = kwargs.get("tensor_parallel_word_embeddings", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)


class BaseModelMixin_TP(PreTrainedModelMixin_TP, BaseModelMixin):
    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        # word embeddings
        self.wte.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wte.")

        # positional embeddings
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "wpe.")
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

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe = Embedding_TP(
                max_position_embeddings,
                self.embed_dim,
                std=self.initializer_range,
                tensor_parallel_word_embeddings=False,
                use_padding_free_transformer=self._use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )
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
