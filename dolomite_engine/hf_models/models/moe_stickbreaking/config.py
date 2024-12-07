from ..moe_dolomite import MoEDolomiteConfig
from ..stickbreaking import StickBreakingConfig


class MoEStickBreakingConfig(MoEDolomiteConfig, StickBreakingConfig):
    model_type = "moe_stickbreaking"
