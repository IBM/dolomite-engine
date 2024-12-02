import torch

from ...enums import PositionEmbeddingType
from ...mixins import PreTrainedModelMixin
from ..gpt_dolomite import GPTDolomiteModel
from .config import DesyncResidualConfig
from .layer import DesyncResidualBlock


class DesyncResidualPreTrainedModel(PreTrainedModelMixin):
    config_class = DesyncResidualConfig
    layer_class = DesyncResidualBlock
    _no_split_modules = ["DesyncResidualBlock"]
    _supports_flash_attn_2 = False


class DesyncResidualModel(DesyncResidualPreTrainedModel, GPTDolomiteModel):
    def __init__(self, config: DesyncResidualConfig, **kwargs) -> None:
        self.tensor_parallel_size = config.pretraining_tensor_parallel_size

        super().__init__(config, **kwargs)

        if self.position_embedding_type == PositionEmbeddingType.alibi:
            raise NotImplementedError("currently DesyncResidual doesn't support alibi")

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
