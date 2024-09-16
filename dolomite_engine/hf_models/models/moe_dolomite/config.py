from ...config import CommonConfig


class MoEDolomiteConfig(CommonConfig):
    model_type = "moe_dolomite"

    def __init__(
        self,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        shared_n_inner: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.shared_n_inner = shared_n_inner

        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
