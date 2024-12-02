from ..gpt_dolomite.config import GPTDolomiteConfig


class StickbreakingConfig(GPTDolomiteConfig):
    model_type = "stickbreaking"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        sb_remainder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sb_remainder = sb_remainder
        self.add_qkv_bias = add_qkv_bias
