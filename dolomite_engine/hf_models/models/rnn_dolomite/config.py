from ...config import CommonConfig


class RNNDolomiteConfig(CommonConfig):
    model_type = "rnn_dolomite"

    def __init__(self, attention_pattern: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.attention_pattern = attention_pattern
        if self.attention_pattern is None:
            self.attention_pattern = "d" * self.n_layer
