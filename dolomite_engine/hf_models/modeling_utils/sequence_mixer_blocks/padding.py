# the following code is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

import torch
import torch.nn.functional as F
from einops import repeat


class _IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert input.dim() >= 2

        ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.size(0)

        other_shape = input.size()[1:]
        indices = indices.unsqueeze(1).expand(-1, other_shape.numel())

        input = input.reshape(input.size(0), -1)
        input = input.gather(0, indices)
        input = input.reshape(-1, *other_shape)

        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        assert grad_output.dim() >= 2

        indices = ctx.saved_tensors[0]
        first_axis_dim = ctx.first_axis_dim

        other_shape = grad_output.size()[1:]
        grad_output = grad_output.view(grad_output.size(0), -1)
        grad_input = torch.zeros(
            (first_axis_dim, grad_output.size(1)), device=grad_output.device, dtype=grad_output.dtype
        )

        indices = indices.unsqueeze(1).expand(-1, grad_output.size(1))

        grad_input.scatter_(0, indices, grad_output)
        grad_input = grad_input.reshape(first_axis_dim, *other_shape)

        return grad_input, None


class _IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
        assert indices.dim() == 1
        assert values.dim() >= 2

        ctx.save_for_backward(indices)
        output = torch.zeros(first_axis_dim, *values.size()[1:], device=values.device, dtype=values.dtype)
        output[indices] = values

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        indices = ctx.saved_tensors[0]
        grad_values = grad_output[indices]
        return grad_values, None, None


def index_first_axis(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return _IndexFirstAxis.apply(input, indices)


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    hidden_states = hidden_states.view(-1, *hidden_states.size()[1:])
    hidden_states = index_first_axis(hidden_states, indices)

    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch


def pad_input(x: torch.Tensor, indices: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    x = _IndexPutFirstAxis.apply(x, indices, batch_size * sequence_length)
    x = x.view(batch_size, sequence_length, *x.size()[1:])
    return x
