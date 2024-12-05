from ..moe_dolomite.layer import SparseMoEBlock
from ..moe_ladder_residual import MoELadderResidualConfig
from ..moe_ladder_residual.layer import MoELadderResidualBlock


class MoELadderResidualBlock_TP(MoELadderResidualBlock):
    def __init__(
        self,
        config: MoELadderResidualConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        moe_implementation: str,
        layer_idx: int | None = None,
    ) -> None:
        SparseMoEBlock.__init__(
            self,
            config=config,
            attention_implementation=attention_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            moe_implementation=moe_implementation,
            layer_idx=layer_idx,
        )
