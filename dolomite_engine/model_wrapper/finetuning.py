import torch

from ..utils import register_profiler, register_timer
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: dict) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        if not self.use_padding_free_transformer:
            for i in batch:
                batch[i] = batch[i].to(self.input_device)

        model_outputs = self.model(**batch)

        loss = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.loss
        return loss
