from typing import Any

from ...utils import BaseArgs


class _MLPArgs(BaseArgs):
    mlp_block_type: str = "MLP"
    intermediate_size: int | None
    activation_function: str
    dropout: float
    add_bias: bool

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_block_type == "MLP"


class _MoEArgs(_MLPArgs):
    mlp_block_type: str = "MoE"
    shared_intermediate_size: int | None
    num_experts: int
    num_experts_per_tok: int

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_block_type == "MoE"
