from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel

from ...distributed import tensor_to_dtensor
from ...utils import ProcessGroupManager


def get_autoregressive_language_modeling_loss_TP(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    upcast_logits_for_loss: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_padding_free_transformer: bool = False,
    reduction: str = "mean",
    tensor_parallel_word_embeddings: bool = False,
) -> DTensor:
    if use_padding_free_transformer:
        shift_logits = lm_logits[:-1, :]
        shift_labels = labels[1:].to(shift_logits.device)

        # this is needed so that the last token of current example doesn't predict first token of next example
        drop_loss_positions = cu_seqlens[1:-1] - 1
        shift_labels[drop_loss_positions] = -100
    else:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

    shift_logits = tensor_to_dtensor(
        shift_logits,
        device_mesh=tp_mesh,
        current_placement=Shard(-1) if tensor_parallel_word_embeddings else Replicate(),
    )
    shift_labels = tensor_to_dtensor(shift_labels, device_mesh=tp_mesh, current_placement=Replicate())

    if upcast_logits_for_loss:
        shift_logits = shift_logits.float()

    loss_context = loss_parallel if tensor_parallel_word_embeddings else nullcontext
    with loss_context():
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction=reduction
        )

    return loss
