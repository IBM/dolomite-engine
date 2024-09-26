from ...config import CommonConfig


class RNNDolomiteConfig(CommonConfig):
    model_type = "rnn_dolomite"

    def __init__(self, attention_pattern: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)

        assert len(attention_pattern) == self.n_layer, "Attention patterns must be specified for each layer"
        self.attention_pattern = attention_pattern
