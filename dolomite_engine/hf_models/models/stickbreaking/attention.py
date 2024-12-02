import math

import torch
import torch.nn
import torch.nn.functional as F
from transformers import DynamicCache

from ...enums import InitMethod
from ...modeling_utils import Attention, ParameterizedLinear
from .config import StickBreakingConfig
from .sb_varlen import sb_attn_varlen_, sb_flash_attn_varlen


@torch.compile
def decoding_stickbreaking(q, k, v, scale=None):
    """
    Stick-breaking attention weights.
    """
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    # logits = q @ k[..., :-1, :].transpose(-1, -2) * scale

    assert q.size(2) == 1
    original_dtype = q.dtype
    q = q.float()
    k = k.float()
    # logits = torch.einsum('bhid,bhjd->bhij', q, k[..., :-1, :]) * scale
    logits = q @ k[..., :-1, :].transpose(-1, -2) * scale
    # logits = logits.float()
    log_z = F.logsigmoid(logits).to(original_dtype)
    log_beta = F.logsigmoid(-logits).to(original_dtype)
    # re_cum_log_beta = log_beta.sum(dim=-1, keepdim=True) - log_beta.cumsum(dim=-1)
    re_cum_log_beta = log_beta.flip(-1).cumsum(dim=-1).flip(-1) - log_beta
    # re_cum_log_beta = log_beta.sum(dim=-1, keepdim=True) - log_beta.cumsum(dim=-1)
    log_att = log_z + re_cum_log_beta
    # print("log_att", log_att[0, 0, 0, -20:])
    att = log_att.exp()
    #  print("att    ", att[0, 0, 0, -20:])
    out = torch.einsum("bhij,bhjd->bhid", att, v[..., :-1, :])
    # out = att @ v[..., :-1, :]
    return out, 1 - att.sum(dim=-1)


class SBAttention(Attention):
    def __init__(self, config: StickBreakingConfig, causal: bool, layer_idx: int | None = None) -> None:
        super().__init__(config, causal, layer_idx)
        self.sb_remainder = config.sb_remainder
        if config.add_qkv_bias:
            init_method = InitMethod(config.init_method)
            initializer_range = config.initializer_range
            m_width = config.m_width
            std = initializer_range
            if init_method == InitMethod.mup:
                std /= math.sqrt(m_width)
            self.c_attn = ParameterizedLinear(
                self.hidden_size,
                self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
                bias=True,
                std=std,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sb_metadata=None,
    ) -> torch.Tensor:
        assert past_key_values is not None
        query, key, value = self._prepare_qkv_for_forward(hidden_states)
        softmax_scale = self._get_softmax_scale()
        self.attn_pdrop if self.training else 0
        key, value = past_key_values.update(key, value, self.layer_idx)
        bsz_, _, length_, _ = query.size()

        q_ = query
        k_ = key
        v_ = value

        def sb_attn(q_, k_, v_):
            bsz, heads, length, hdim = k_.size()
            q_ = q_.flatten(0, 1)
            k_ = k_.flatten(0, 1)
            v_ = v_.flatten(0, 1)
            attn_output, rem = sb_flash_attn_varlen(
                q=q_,
                k=k_,
                v=v_,
                cu_seqlens=torch.arange(bsz + 1, dtype=torch.int32, device=query.device) * length,
                inv_temp=softmax_scale,
                zero_start=True,
            )
            if self.sb_remainder:
                attn_output = attn_output.add_(rem[..., None] * v_)
            attn_output = attn_output.view(bsz, heads, length, hdim)
            return attn_output, rem

        if query.size(2) == key.size(2):
            attn_output, rem = sb_attn(q_, k_, v_)
        else:
            attn_output, rem = decoding_stickbreaking(q=q_, k=k_, v=v_, scale=softmax_scale)
            if self.sb_remainder:
                attn_output = attn_output + rem[..., None] * v_[..., -1:, :]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()

        # ==========================================================================================
        # attn_output -> (total_q, num_heads, head_dim)
        # ==========================================================================================
        attn_output = attn_output.view(bsz_, length_, self.hidden_size)

        # ==========================================================================================
        # attn_output -> (total_q, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class PaddingFreeSBAttention(SBAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sb_metadata=None,
    ) -> torch.Tensor:
        assert past_key_values is None

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        softmax_scale = self._get_softmax_scale()
        self.attn_pdrop if self.training else 0

        cu_row_blocks, first_row_block, sequence_ids = sb_metadata
        value = value.permute(1, 0, 2)
        attn_output, rem = sb_attn_varlen_(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=value,
            inv_temp=softmax_scale,
            cu_seqlens=cu_seqlens[1:],
            first_row_block=first_row_block,
            cu_row_blocks=cu_row_blocks,
            sequence_ids=sequence_ids,
        )
        if self.sb_remainder:
            attn_output = attn_output + rem[..., None] * v_
        attn_output = attn_output.permute(1, 0, 2)

        attn_output = attn_output.view(-1, self.hidden_size)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)
        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(total_q, -1, self.head_dim)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(total_q, self.num_heads, -1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value
