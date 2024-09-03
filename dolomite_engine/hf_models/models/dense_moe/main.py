from ...mixins import CausalLMModelMixin
from .base import DenseMoEModel, DenseMoEPreTrainedModel
from .config import DenseMoEConfig


class DenseMoEForCausalLM(DenseMoEPreTrainedModel, CausalLMModelMixin):
    base_model_class = DenseMoEModel

    def __init__(self, config: DenseMoEConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
