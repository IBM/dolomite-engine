from ..moe_dolomite.config import MoEDolomiteConfig
from ..stickbreaking.config import StickBreakingConfig


class MoEStickBreakingConfig(StickBreakingConfig, MoEDolomiteConfig):
    model_type = "moe_stickbreaking"
