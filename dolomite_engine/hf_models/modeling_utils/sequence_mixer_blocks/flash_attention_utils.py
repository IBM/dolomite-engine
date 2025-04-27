import torch

from ....utils import is_flash_attention_available


if is_flash_attention_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_length: int,
    causal: bool,
    dropout: float = 0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    softcap: float = 0,
) -> torch.Tensor:
    window_size = (-1, -1)
    if sliding_window is not None and key.size(1) > sliding_window:
        window_size = (sliding_window, sliding_window)

    # Contains at least one padding token in the sequence
    if attention_mask is None:
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
    else:
        batch_size = query.shape[0]
        query, key, value, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query, key, value, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
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
