import torch
import torch.distributed

from ..communication import Communication
from ..utils import ProcessGroupManager
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    def forward(self, batch: dict) -> dict:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        if ProcessGroupManager.is_tensor_parallel_enabled():
            batch = self._broadcast_inputs_for_tensor_parallel(batch)

        model_outputs = self.model(**batch)
        loss = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.loss

        return {"loss": loss}

    def _broadcast_inputs_for_tensor_parallel(self, batch: dict) -> dict:
        device = torch.cuda.current_device()

        is_tp_first_rank = ProcessGroupManager.is_tensor_parallel_first_rank()
        tp_source_rank = ProcessGroupManager.get_tensor_parallel_first_rank()
        tp_group = ProcessGroupManager.get_tensor_parallel_group()

        if self.use_padding_free_transformer:
            keys = ["input_ids", "position_ids", "labels", "cu_seqlens", "max_seqlen"]

            if is_tp_first_rank:
                metadata = torch.tensor([batch["cu_seqlens"].numel(), batch["input_ids"].numel()], device=device)
            else:
                metadata = torch.empty(2, dtype=torch.long, device=device)

            torch.distributed.broadcast(metadata, src=tp_source_rank, group=tp_group)
            cu_seqlens_num_elements, input_ids_num_elements = metadata

            if not is_tp_first_rank:
                batch = {
                    "input_ids": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "position_ids": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "labels": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "cu_seqlens": torch.empty(cu_seqlens_num_elements, dtype=torch.int32, device=device),
                    "max_seqlen": torch.empty(1, dtype=torch.long, device=device),
                }
        else:
            keys = ["input_ids", "attention_mask", "labels"]

            batch_shape = batch["input_ids"].shape if is_tp_first_rank else None
            batch_shape = Communication.broadcast_object(batch_shape, src=tp_source_rank, group=tp_group)

            if not is_tp_first_rank:
                batch = {key: torch.empty(batch_shape, dtype=torch.long, device=device) for key in keys}

        for key in keys:
            torch.distributed.broadcast(batch[key], src=tp_source_rank, group=tp_group)

        return batch
