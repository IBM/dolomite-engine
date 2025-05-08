import torch

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_flash_attention_2_available, is_flash_attention_3_available
from .padding import (
    _get_unpad_data,
    compute_cu_seqlens_and_max_seqlen_from_attention_mask,
    index_first_axis,
    pad_input,
)


if is_flash_attention_2_available():
    from flash_attn import flash_attn_func as flash_attention_2
    from flash_attn import flash_attn_varlen_func as flash_attention_2_varlen

if is_flash_attention_3_available():
    from flash_attn_interface import flash_attn_func as flash_attention_3
    from flash_attn_interface import flash_attn_varlen_func as flash_attention_3_varlen


def _upad_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor]]:
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key.size()

    key = index_first_axis(key.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value = index_first_axis(value.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)

    if query_length == kv_seq_len:
        query = index_first_axis(query.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query = query.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]

        indices_q, cu_seqlens_q, max_seqlen_in_batch_q = _get_unpad_data(attention_mask)
        max_seqlen_in_batch_q = max_seqlen_in_batch_q.item()

        query = query.view(-1, *query.size()[1:])
        query = index_first_axis(query, indices_q)

    return query, key, value, indices_q, cu_seqlens_q, cu_seqlens_k, max_seqlen_in_batch_q, max_seqlen_in_batch_k


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: torch.Tensor | None,
    use_padding_free_transformer: bool,
    query_length: int,
    causal: bool,
    dropout: float = 0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    softcap: float = 0,
) -> torch.Tensor:
    use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
    use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)

    if use_flash_attention_3:
        assert dropout == 0

    assert use_flash_attention_3 or use_flash_attention_2, "enable flash_attention_2 or flash_attention_3"

    if use_padding_free_transformer:
        assert use_flash_attention_3 or use_flash_attention_2

    window_size = (-1, -1)
    if sliding_window is not None and key.size(1) > sliding_window:
        window_size = (sliding_window, sliding_window)

    if use_padding_free_transformer:
        if use_flash_attention_3:
            attn_output, _ = flash_attention_3_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            attn_output = flash_attention_2_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
    else:
        if attention_mask is None:
            if use_flash_attention_3:
                attn_output, _ = flash_attention_3(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                )
            else:
                attn_output = flash_attention_2(
                    q=query,
                    k=key,
                    v=value,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                )
        else:
            batch_size = query.size(0)

            query, key, value, indices_q, cu_seqlens_q, cu_seqlens_k, max_seqlen_in_batch_q, max_seqlen_in_batch_k = (
                _upad_input(query, key, value, attention_mask, query_length)
            )

            if use_flash_attention_3:
                attn_output, _ = flash_attention_3_varlen(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                )
            else:
                attn_output = flash_attention_2_varlen(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                )

            attn_output = pad_input(attn_output, indices_q, batch_size, query_length)

    return attn_output
