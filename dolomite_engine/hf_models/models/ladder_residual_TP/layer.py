from ..gpt_dolomite_TP.layer import GPTDolomiteBlock_TP
from ..ladder_residual import LadderResidualConfig
from ..ladder_residual.layer import LadderResidualBlock


class LadderResidualBlock_TP(LadderResidualBlock):
    def __init__(
        self,
        config: LadderResidualConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        GPTDolomiteBlock_TP.__init__(
            self,
            config=config,
            attention_implementation=attention_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
        )
