import torch
from transformers import DynamicCache

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

    def forward(
        self,
        previous_attention_out: torch.Tensor,
        previous_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if previous_attention_out is not None:
            if self.m_residual is not None:
                previous_attention_out = previous_attention_out * self.m_residual

            residual = residual + previous_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self.attn(
            current_attention_out,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if previous_mlp_out is not None:
            if self.m_residual is not None:
                previous_mlp_out = previous_mlp_out * self.m_residual

            residual = residual + previous_mlp_out

        current_mlp_out = self.ln_2(residual)
        current_mlp_out = self.mlp(current_mlp_out)

        return current_attention_out, current_mlp_out, residual
