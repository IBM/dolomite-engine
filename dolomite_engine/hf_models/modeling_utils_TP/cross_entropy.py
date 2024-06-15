# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch
import torch.distributed

from ...utils import ProcessGroupManager
from .embedding import get_tensor_parallel_vocab_info


class TensorParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, lm_logits: torch.Tensor, labels: torch.Tensor, vocab_size: int, upcast_logits_for_loss: bool
    ) -> torch.Tensor:
        tp_group = ProcessGroupManager.get_tensor_parallel_group()

        # Maximum value along vocab dimension across all GPUs.
        logits_max = lm_logits.max(dim=-1)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank = get_tensor_parallel_vocab_info(
            vocab_size
        )

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels - vocab_start_index
        masked_labels[labels_mask] = 0

        # Subtract the maximum value.
        lm_logits = lm_logits - logits_max.unsqueeze(dim=-1)

        logits_2d = lm_logits.view(-1, vocab_size_per_tensor_parallel_rank)
        masked_labels = masked_labels.view(-1)

        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_labels]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        if upcast_logits_for_loss:
            lm_logits = lm_logits.float()

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = lm_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize
        exp_logits = exp_logits / sum_exp_logits.unsqueeze(dim=-1)

        ctx.save_for_backward(exp_logits, labels_mask, masked_labels)

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # Retreive tensors from the forward path.
        softmax, labels_mask, masked_labels = ctx.saved_tensors

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = softmax.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1 - labels_mask.view(-1).float()

        grad_2d[arange_1d, masked_labels] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input = softmax * grad_output.unsqueeze(dim=-1)

        return grad_input, None, None, None
