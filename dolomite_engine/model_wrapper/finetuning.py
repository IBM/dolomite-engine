from typing import Union

import torch

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    def __init__(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode):
        super().__init__(args, mode)

        assert not self.reset_attention_mask, "reset_attention_mask is only supported with pretraining"
        assert not self.reset_position_ids, "reset_position_ids is only supported with pretraining"

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
