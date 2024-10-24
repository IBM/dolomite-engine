import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import DynamicCache

from ...enums import AttentionHeadType, InitMethod, PositionEmbeddingType
from ...modeling_utils import get_attention_module, get_normalization_function
from ...modeling_utils.attention import Attention
from ...modeling_utils.linear import ParameterizedLinear
from ..gpt_dolomite.layer import GPTDolomiteBlock
from ..gpt_dolomite.mlp import MLP
from .config import SBDolomiteConfig
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
    def __init__(self, config: SBDolomiteConfig, causal: bool, layer_idx: int | None = None) -> None:
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
            if True or q_.numel() < 2**32:

                attn_output, rem = sb_attn(q_, k_, v_)
            else:
                attn_output_list = []
                rem_list = []
                for i in range(q_.size(1)):
                    attn_output_i, rem_i = sb_attn(q_[:, i : i + 1], k_[:, i : i + 1], v_[:, i : i + 1])
                    attn_output_list.append(attn_output_i)
                    rem_list.append(rem_i)
                attn_output = torch.cat(attn_output_list, dim=1)
                rem = torch.cat(rem_list, dim=1)

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

        # ==========================================================================================
        # hidden_states -> (total_q, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (total_q, num_heads, head_dim)
        # key -> (total_q, num_key_value_heads, head_dim)
        # value -> (total_q, num_key_value_heads, head_dim)
        # ==========================================================================================

        # if self.position_embedding_type == PositionEmbeddingType.rope:
        #     query = apply_rotary_pos_emb(query, rope_cos_sin)
        #     key = apply_rotary_pos_emb(key, rope_cos_sin)

        # ==========================================================================================
        # query -> (total_q, num_heads, head_dim)
        # key -> (total_q, num_key_value_heads, head_dim)
        # value -> (total_q, num_key_value_heads, head_dim)
        # ==========================================================================================

        softmax_scale = self._get_softmax_scale()
        self.attn_pdrop if self.training else 0

        cu_row_blocks, first_row_block, sequence_ids = sb_metadata
        v_ = value.permute(1, 0, 2)
        attn_output, rem = sb_attn_varlen_(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=v_,
            inv_temp=softmax_scale,
            cu_seqlens=cu_seqlens[1:],
            first_row_block=first_row_block,
            cu_row_blocks=cu_row_blocks,
            sequence_ids=sequence_ids,
        )
        if self.sb_remainder:
            attn_output = attn_output + rem[..., None] * v_
        attn_output = attn_output.permute(1, 0, 2)

        # ==========================================================================================
        # attn_output -> (total_q, num_heads, head_dim)
        # ==========================================================================================

        attn_output = attn_output.view(-1, self.hidden_size)

        # ==========================================================================================
        # attn_output -> (total_q, num_heads * head_dim)
        # ==========================================================================================

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


class SBDolomiteBlock(GPTDolomiteBlock):
    """
    Layer implementation for the transformer block
    """

    def __init__(
        self,
        config: SBDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        # self.attn = get_attention_module(
        #     config, True, attention_implementation, use_padding_free_transformer, layer_idx
        # )
        if use_padding_free_transformer:
            self.attn = PaddingFreeSBAttention(config, causal=True, layer_idx=layer_idx)
        else:
            self.attn = SBAttention(config, causal=True, layer_idx=layer_idx)

        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )

        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sb_metadata=None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sb_metadata=sb_metadata,
        )

        if self.m_residual is not None:
            if self.training:
                attn_output = attn_output * self.m_residual
            else:
                attn_output.mul_(self.m_residual)

        # residual connection
        if self.training:
            hidden_states = attn_output + residual
            residual = hidden_states
        else:
            hidden_states = residual.add_(attn_output)

        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.m_residual is not None:
            if self.training:
                feed_forward_hidden_states = feed_forward_hidden_states * self.m_residual
            else:
                feed_forward_hidden_states.mul_(self.m_residual)

        # residual connection
        if self.training:
            hidden_states = residual + feed_forward_hidden_states
        else:
            hidden_states = residual.add_(feed_forward_hidden_states)

        return hidden_states
