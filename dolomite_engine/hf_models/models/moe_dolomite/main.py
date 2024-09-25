from ...mixins import CausalLMMoEModelMixin
from .base import MoEDolomiteModel, MoEDolomitePreTrainedModel
from .config import MoEDolomiteConfig


class MoEDolomiteForCausalLM(MoEDolomitePreTrainedModel, CausalLMMoEModelMixin):
    base_model_class = MoEDolomiteModel

    def __init__(self, config: MoEDolomiteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
