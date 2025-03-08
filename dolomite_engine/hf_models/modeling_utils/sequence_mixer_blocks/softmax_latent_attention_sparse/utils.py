import torch


def interleave_query_key_value_tensor_for_mha(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    interleaved = []
    for i in range(num_heads):
        start_index = i * head_dim
        end_index = start_index + head_dim

        interleaved.append(query_weight[start_index:end_index])
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_mha(
    query_key_value_weight: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_heads, -1)

    query_weight, key_weight, value_weight = query_key_value_weight.chunk(3, -1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_gqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    query_heads_per_group = num_heads // num_key_value_heads

    interleaved = []
    for i in range(num_key_value_heads):
        start_index = i * query_heads_per_group * head_dim
        end_index = start_index + query_heads_per_group * head_dim
        interleaved.append(query_weight[start_index:end_index])

        start_index = i * head_dim
        end_index = start_index + head_dim
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_gqa(
    query_key_value_weight: torch.Tensor, num_heads: int, num_key_value_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_heads_per_group = num_heads // num_key_value_heads
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_key_value_heads, (query_heads_per_group + 2), -1)

    query_weight, key_weight, value_weight = query_key_value_weight.split((query_heads_per_group, 1, 1), 1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


def interleave_query_key_value_tensor_for_mqa(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    # [:] for converting slice to tensor
    return torch.cat([query_weight[:], key_weight[:], value_weight[:]])


def split_query_key_value_tensor_for_mqa(
    query_key_value_weight: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return query_key_value_weight.split((num_heads * head_dim, head_dim, head_dim))


def repeat_key_value(x: torch.Tensor, num_heads: int, num_key_value_heads: int) -> torch.Tensor:
    num_groups = num_heads // num_key_value_heads

    if num_groups == 1:
        return x

    if num_key_value_heads == 1:
        return x.expand(-1, num_heads, -1, -1)

    return x.repeat_interleave(num_groups, dim=1)
