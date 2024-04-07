import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from ..utils import get_module_class_from_name


def block_checkpointing(model: nn.Module, block_name: str, checkpoint_every: int = 1) -> None:
    block_class = get_module_class_from_name(model, block_name)
    block_idx = 0

    def _whether_to_checkpoint(submodule: nn.Module) -> bool:
        nonlocal block_idx

        if isinstance(submodule, block_class):
            block_idx += 1
            if (block_idx - 1) % checkpoint_every == 0:
                return True
        return False

    apply_activation_checkpointing(model, check_fn=_whether_to_checkpoint)
