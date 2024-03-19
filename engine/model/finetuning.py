from typing import List, Tuple

import torch

from ..utils import register_profiler, register_timer
from .base import Model


class ModelForFinetuning(Model):
    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: Tuple[List[int]]) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        batch = self.prepare_batch(batch)

        if not self.use_padding_free_transformer:
            for i in batch:
                batch[i] = batch[i].to(self.input_device)

        model_outputs = self.model(**batch)

        if type(model_outputs) is tuple:
            loss = model_outputs[0]
        else:
            loss = model_outputs.loss

        return loss
