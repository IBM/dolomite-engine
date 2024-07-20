from ...config import CommonConfig


class RNNDolomiteConfig(CommonConfig):
    model_type = "rnn_dolomite"

    def __init__(self, attention_patterns: str | None = None, use_short_conv: bool = True, use_gate: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        assert len(attention_patterns) == self.n_layer, "Attention patterns must be specified for each layer"
        self.attention_patterns = attention_patterns
        self.use_short_conv = use_short_conv
        self.use_gate = use_gate
