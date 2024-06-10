from typing import Union

import torch
import torch.nn.functional as F

from dolomite_engine.enums import Mode

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from .base import ModelWrapper


class ModelWrapperForPretraining(ModelWrapper):
    def __init__(self, args: TrainingArgs | InferenceArgs | ExportArgs, mode: Mode):
        super().__init__(args, mode)

        self.upcast_logits_for_loss = getattr(self.config, "upcast_logits_for_loss", False)

    def forward(self, batch: dict) -> torch.Tensor:
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
            tokens = tokens.to(torch.cuda.current_device())

        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]

        if self.use_padding_free_transformer:
            input_ids, cu_seqlens, max_seqlen, position_ids = self._get_padding_free_inputs(input_ids)

            model_outputs = self.model(
                input_ids=input_ids, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            model_outputs = self.model(input_ids=input_ids)

        logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.logits
        if self.upcast_logits_for_loss:
            logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

        return loss

    def _setup_model(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
        super()._setup_model(args)

        assert not self.is_encoder_decoder, "currently encoder_decoder models are not supported for pretraining"

        if self.use_padding_free_transformer:
            batch_size = args.training_parameters.micro_batch_size
            sequence_length = args.datasets[0].class_args.get("sequence_length")

            if not self.reset_attention_mask:
                self.register_buffer(
                    "cu_seqlens",
                    torch.arange(
                        0,
                        batch_size * sequence_length + 1,
                        sequence_length,
                        dtype=torch.int32,
                        device=torch.cuda.current_device(),
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "max_seqlen", torch.tensor(sequence_length, device=torch.cuda.current_device()), persistent=False
                )

            if self.reset_position_ids:
                assert self.reset_attention_mask, "reset_attention_mask should be specified with reset_position_ids"
            else:
                self.register_buffer(
                    "position_ids",
                    torch.arange(0, sequence_length, 1, device=torch.cuda.current_device()).repeat(batch_size),
                    persistent=False,
                )
        else:
            assert (
                not self.reset_attention_mask
            ), "currently reset_attention_mask is only implemented for padding free transformer"
            assert (
                not self.reset_position_ids
            ), "currently reset_position_ids is only implemented for padding free transformer"

    def _get_padding_free_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        input_ids = input_ids.reshape(-1)

        if self.reset_attention_mask:
            num_tokens_in_batch = batch_size * sequence_length

            document_end_positions = input_ids == self.eos_token_id
            for i in range(sequence_length - 1, num_tokens_in_batch, sequence_length):
                document_end_positions[i] = 1
            cu_seqlens = document_end_positions.nonzero(as_tuple=True)[0] + 1
            cu_seqlens = torch.cat([torch.tensor([0], device=input_ids.device), cu_seqlens])
            cu_seqlens = cu_seqlens.to(torch.int32)

            seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seqlen.max()

            if self.reset_position_ids:
                position_ids = torch.cat(
                    [torch.arange(0, i, 1, dtype=torch.int32, device=input_ids.device) for i in seqlen]
                )
            else:
                position_ids = self.position_ids
        else:
            cu_seqlens = self.cu_seqlens
            max_seqlen = self.max_seqlen
            position_ids = self.position_ids

        return input_ids, cu_seqlens, max_seqlen, position_ids
