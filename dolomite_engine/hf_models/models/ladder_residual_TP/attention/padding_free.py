import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Partial, Replicate
from transformers import DynamicCache

from .....distributed import dtensor_to_tensor, tensor_to_dtensor
from .....utils import ProcessGroupManager, is_flash_attention_available
from ....enums import AttentionHeadType, PositionEmbeddingType
from ....modeling_utils import apply_rotary_pos_emb
from ....modeling_utils_TP import get_module_placements
from ....modeling_utils_TP.attention import PaddingFreeAttention_TP
from ...ladder_residual import LadderResidualConfig


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class LadderResidualPaddingFreeAttention_TP(PaddingFreeAttention_TP):
    def __init__(
        self, config: LadderResidualConfig, causal: bool, layer_idx: int = None, sequence_parallel: bool = False
    ) -> None:
        super().__init__(config, causal, layer_idx, sequence_parallel)
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.placement = get_module_placements(True, sequence_parallel)

    def forward(
        self,
        current_attention_out: torch.Tensor,
        current_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert past_key_values is None

        query, key, value = self._prepare_qkv_for_forward(current_attention_out)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        softmax_scale = self._get_softmax_scale()
        dropout_p = self.attn_pdrop if self.training else 0

        current_attention_out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=self.causal,
        )

        current_attention_out = current_attention_out.view(-1, self.hidden_size)

        current_attention_out = F.linear(
            current_attention_out, dtensor_to_tensor(self.c_proj.weight), dtensor_to_tensor(self.c_proj.bias)
        )
        current_attention_out = dtensor_to_tensor(
            tensor_to_dtensor(
                current_attention_out,
                device_mesh=self.tp_mesh,
                current_placement=Partial(),
                desired_placement=self.placement,
            )
        )
        current_attention_out = self.resid_dropout(current_attention_out)

        return current_attention_out, current_mlp_out, residual

    def _prepare_qkv_for_forward(
        self, current_attention_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # the output of following is a tuple if using MQA with tensor parallel
        current_attention_out = dtensor_to_tensor(
            tensor_to_dtensor(
                current_attention_out,
                device_mesh=self.tp_mesh,
                current_placement=self.placement,
                desired_placement=Replicate(),
            )
        )
        current_attention_out = F.linear(
            current_attention_out, dtensor_to_tensor(self.c_attn.weight), dtensor_to_tensor(self.c_attn.bias)
        )

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(current_attention_out)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(current_attention_out)
        elif self.attention_head_type == AttentionHeadType.mqa:
            query, key, value = self._prepare_qkv_for_forward_mqa(current_attention_out)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        return query, key, value
