import torch


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

    return input_tensor


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def _apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states
