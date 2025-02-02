from ...config import CommonConfig


class MoEDolomiteConfig(CommonConfig):
    model_type = "moe_dolomite"

    def __init__(
        self,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.001,
        shared_n_inner: int | None = None,
        use_aux_free_moe: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.shared_n_inner = shared_n_inner

        self.router_aux_loss_coef = router_aux_loss_coef

        self.use_aux_free_moe = use_aux_free_moe
