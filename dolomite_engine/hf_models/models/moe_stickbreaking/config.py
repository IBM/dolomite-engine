from ..stickbreaking.config import StickBreakingConfig
from ..moe_dolomite.config import MoEDolomiteConfig

class MoEStickBreakingConfig(StickBreakingConfig,MoEDolomiteConfig):
    model_type = "moe_stickbreaking"