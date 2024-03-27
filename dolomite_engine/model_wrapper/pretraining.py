from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..utils import register_profiler, register_timer
from .base import ModelWrapper


class ModelWrapperForPretraining(ModelWrapper):
    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: Tuple[List[int]]) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        # for pretraining we compute loss externally here instead of relying on transformers.
        # this is done because megatron's dataset returns batches of length (sequence_length + 1)
        # instead of (sequence_length), so we need to trim the input_ids before forward pass.
        # transformers does forward pass before however and then trims the tokens.

        tokens: torch.Tensor = batch["text"]
        if not tokens.is_cuda:
            tokens = tokens.to(self.input_device)

        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        if self.use_padding_free_transformer:
            model_outputs = self.model(
                input_ids=input_ids.reshape(-1),
                position_ids=self.position_ids,
                cu_seqlens=self.cu_seqlens,
                max_seqlen=self.max_seqlen,
            )
        else:
            model_outputs = self.model(input_ids=input_ids)

        if type(model_outputs) is tuple:
            logits = model_outputs[0]
        else:
            logits = model_outputs.logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

        return loss

    def _setup_model(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
        super()._setup_model(args)

        assert not self.is_encoder_decoder, "currently encoder_decoder models are not supported for pretraining"

        if self.use_padding_free_transformer:
            batch_size = args.training_parameters.batch_size_per_gpu
            sequence_length = args.datasets[0].class_args.get("sequence_length")

            self.register_buffer(
                "cu_seqlens",
                torch.arange(
                    0, batch_size * sequence_length + 1, sequence_length, dtype=torch.int32, device=self.input_device
                ),
                persistent=False,
            )
            self.register_buffer(
                "max_seqlen", torch.tensor(sequence_length, device=self.input_device), persistent=False
            )
            self.register_buffer(
                "position_ids",
                torch.arange(0, sequence_length, 1, device=self.input_device).repeat(batch_size),
                persistent=False,
            )
