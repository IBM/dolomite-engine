# from ..moe_dolomite.config import MoEDolomiteConfig
# from ..stickbreaking.config import StickBreakingConfig
from ...config import CommonConfig


class MoEStickBreakingConfig(CommonConfig):
    model_type = "moe_stickbreaking"

    def __init__(
        self,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        shared_n_inner: int | None = None,
        add_qkv_bias: bool = False,
        sb_remainder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sb_remainder = sb_remainder
        self.add_qkv_bias = add_qkv_bias

        if add_qkv_bias:
            assert not self.add_bias

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.shared_n_inner = shared_n_inner

        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

