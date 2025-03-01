from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel

from ..distributed import tensor_to_dtensor
from ..enums import Kernel
from ..kernels import is_kernel_allowed
from ..utils import ProcessGroupManager, is_cute_kernels_available


if is_cute_kernels_available():
    from cute_kernels import cross_entropy_cute, fused_linear_cross_entropy_cute


def get_autoregressive_language_modeling_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
    vocab_weight: torch.Tensor | None = None,
    logits_multiplier: float = 1,
    cu_seqlens: torch.Tensor | None = None,
    use_padding_free_transformer: bool = False,
    reduction: str = "mean",
    shift_logits_and_labels: bool = True,
    tensor_parallel_enabled: bool = False,
) -> torch.Tensor | DTensor:
    if shift_logits_and_labels:
        lm_logits = lm_logits[..., :-1, :]
        labels = labels[..., 1:]

    if use_padding_free_transformer:
        if shift_logits_and_labels:
            assert cu_seqlens is not None

            # this is needed so that the last token of current example doesn't predict first token of next example
            drop_loss_positions = cu_seqlens[1:-1] - 1
            labels[drop_loss_positions] = -100
    else:
        assert cu_seqlens is None

    if is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute):
        assert lm_logits is None
        assert not tensor_parallel_enabled

        loss = fused_linear_cross_entropy_cute(
            x=hidden_states,
            weight=vocab_weight,
            labels=labels,
            reduction=reduction,
            logits_multiplier=logits_multiplier,
        )
    elif is_kernel_allowed(Kernel.cross_entropy_cute):
        assert hidden_states is None
        assert vocab_weight is None
        assert not tensor_parallel_enabled

        loss = cross_entropy_cute(
            x=lm_logits.reshape(-1, lm_logits.size(-1)),
            labels=labels.reshape(-1),
            reduction=reduction,
            logits_multiplier=logits_multiplier,
        )
    else:
        assert logits_multiplier == 1
        loss_context = nullcontext

        if tensor_parallel_enabled:
            loss_context = loss_parallel
            tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

            lm_logits = tensor_to_dtensor(lm_logits, device_mesh=tp_mesh, current_placement=Shard(-1))
            labels = tensor_to_dtensor(labels, device_mesh=tp_mesh, current_placement=Replicate())

        lm_logits = lm_logits.float()

        with loss_context():
            loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1), reduction=reduction)

    return loss


_AUX_LOSS: torch.Tensor | float = 0


def clear_aux_loss() -> None:
    global _AUX_LOSS
    _AUX_LOSS = 0


def add_aux_loss(aux_loss: torch.Tensor) -> None:
    global _AUX_LOSS
    _AUX_LOSS = _AUX_LOSS + aux_loss


def get_aux_loss() -> torch.Tensor:
    return _AUX_LOSS
