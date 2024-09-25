from ...mixins import CausalLMModelMixin
from .base import GPTCrossLayerModel, GPTCrossLayerPreTrainedModel


class GPTCrossLayerForCausalLM(GPTCrossLayerPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTCrossLayerModel

    def get_global_local_idx(self, index: int) -> tuple[int, int]:
        return self.transformer.layer_map[index]
