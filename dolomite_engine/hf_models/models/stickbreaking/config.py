from ...config import CommonConfig


class StickBreakingConfig(CommonConfig):
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

        if add_qkv_bias:
            assert not self.add_bias
