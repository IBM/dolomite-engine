from ...config import CommonConfig
from ..dense import CausalLMModelMixin


class CausalLMMoEModelMixin(CausalLMModelMixin):
    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
