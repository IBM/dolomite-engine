from ..moe_dolomite import MoEDolomiteConfig
from ..rnn_dolomite import RNNDolomiteConfig


class RNNMoEDolomiteConfig(RNNDolomiteConfig, MoEDolomiteConfig):
    model_type = "rnn_moe_dolomite"
