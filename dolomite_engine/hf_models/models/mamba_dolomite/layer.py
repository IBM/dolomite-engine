import torch
from transformers import DynamicCache

from ...modeling_utils import get_normalization_function
from ..gpt_dolomite.layer import GPTDolomiteBlock
from .config import Mamba2DolomiteConfig


class Mamba2DolomiteBlock(GPTDolomiteBlock):
    def __init__(
        self,
        config: Mamba2DolomiteConfig,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        if self.is_attention_layer:
            super().__init__(
                config=config,
                attention_implementation=attention_implementation,
                use_padding_free_transformer=use_padding_free_transformer,
                layer_idx=layer_idx,
            )
        else:
            pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if self.is_attention_layer:
            hidden_states = super().forward(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            pass

        return hidden_states
