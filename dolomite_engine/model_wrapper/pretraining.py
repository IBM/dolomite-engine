from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..enums import AttentionImplementation, Mode, MoEImplementation
from ..hf_models.modeling_utils_TP import tensor_to_dtensor
from ..utils import ProcessGroupManager
from .base import ModelWrapper


class ModelWrapperForPretraining(ModelWrapper):
    def __init__(
        self,
        mode: Mode,
        model_name: str | None,
        pretrained_config: dict | None,
        model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        dtype: torch.dtype,
        efficient_initialization: bool,
        attention_implementation: AttentionImplementation,
        moe_implementation: MoEImplementation,
        use_padding_free_transformer: bool,
        tensor_parallel_word_embeddings: bool,
        sequence_parallel: bool,
        micro_batch_size: int,
        sequence_length: int,
        num_pipeline_stages: int,
        pipeline_stage_id: int,
        neft_alpha: float | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        reset_attention_mask: bool = False,
        reset_position_ids: bool = False,
    ) -> None:
        """initializes a model wrapper for a HuggingFace model

        Args:
            mode (Mode): training / inference mode
            model_name (str | None): path of the model on disk or HF hub
            pretrained_config (dict | None): config of the model to load model from, only used if `model_name` is None
            model_class (AutoModelForCausalLM | AutoModelForSeq2SeqLM): HF model class to use for model loading
            dtype (torch.dtype): dtype for the model
            efficient_initialization (bool): whether to use efficient initialization for the model initialization, saves CPU memory
            attention_implementation (AttentionImplementation): attention implementation for the model
            use_padding_free_transformer (bool): whether to use padding free transformer
            tensor_parallel_word_embeddings (bool): whether to use tensor parallel word embeddings
            sequence_parallel (bool): whether to use sequence parallel
            micro_batch_size (int): micro batch size for pretraining
            sequence_length (int): sequence length for pretraining
            num_pipeline_stages (int): number of stages for the pipeline
            pipeline_stage_id (int): current pipeline stage id
            neft_alpha (float | None, optional): alpha parameter for NEFTune. Defaults to None.
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            reset_attention_mask (bool, optional): whether to reset attention mask during pretraining. Defaults to False.
            reset_position_ids (bool, optional): whether to reset position ids during pretraining. Defaults to False.
        """

        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.reset_attention_mask = reset_attention_mask
        self.reset_position_ids = reset_position_ids

        super().__init__(
            mode=mode,
            model_name=model_name,
            pretrained_config=pretrained_config,
            model_class=model_class,
            dtype=dtype,
            efficient_initialization=efficient_initialization,
            attention_implementation=attention_implementation,
            moe_implementation=moe_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            sequence_parallel=sequence_parallel,
            num_pipeline_stages=num_pipeline_stages,
            pipeline_stage_id=pipeline_stage_id,
            neft_alpha=neft_alpha,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
        )

        if self.is_pipeline_parallel_enabled:
            assert not self.reset_attention_mask, "reset_attention_mask is not supported with pipeline parallelism"
            assert not self.reset_position_ids, "reset_position_ids is not supported with pipeline parallelism"

    def forward(self, batch: dict) -> dict:
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

        if isinstance(batch, torch.Tensor):
            batch = {"text": batch}

        input_ids, labels = self._prepare_inputs_ids_and_labels_for_forward(batch)
        batch = self._prepare_model_inputs(input_ids)

        model_outputs = self.model(**batch, return_dict=True)

        # without pipeline parallel, we compute the loss outside
        if not self.is_pipeline_parallel_enabled:
            model_outputs = self.get_loss(model_outputs, labels)

        return model_outputs

    def get_loss(self, model_outputs, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(model_outputs, torch.Tensor):
            logits = model_outputs
        else:
            logits: torch.Tensor = model_outputs.logits

        if self.upcast_logits_for_loss:
            logits = logits.float()

        loss_context = nullcontext

        if ProcessGroupManager.is_tensor_parallel_enabled():
            logits = tensor_to_dtensor(
                logits,
                device_mesh=self.tp_mesh,
                current_placement=Shard(-1) if self.tensor_parallel_word_embeddings else Replicate(),
            )
            labels = tensor_to_dtensor(labels, device_mesh=self.tp_mesh, current_placement=Replicate())

            if self.tensor_parallel_word_embeddings:
                loss_context = loss_parallel

        with loss_context():
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

        if hasattr(model_outputs, "aux_loss"):
            aux_loss = model_outputs.aux_loss
            loss = lm_loss + self.router_aux_loss_coef * aux_loss

            output = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}
        else:
            loss = lm_loss
            output = {"loss": loss}

        return output

    def broadcast_tensor_parallel_input(self, tokens: dict, shape: tuple[int]) -> torch.Tensor:
        if ProcessGroupManager.is_tensor_parallel_first_rank():
            tokens = tokens.to(torch.cuda.current_device())
        else:
            tokens = torch.empty(shape, dtype=torch.long, device=torch.cuda.current_device())

        torch.distributed.broadcast(
            tokens,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )

        return tokens

    def _prepare_model_inputs(self, input_ids: torch.Tensor) -> dict:
        batch = {}

        if self.use_padding_free_transformer:
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
                # we move to CPU here otherwise FlashAttention will move to CPU on every invocation i.e all layers
                max_seqlen = seqlen.max().item()

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

            batch["cu_seqlens"] = cu_seqlens
            batch["max_seqlen"] = max_seqlen
            batch["position_ids"] = position_ids

        batch["input_ids"] = input_ids

        if ProcessGroupManager.is_tensor_parallel_enabled():
            batch["output_parallel_lm_logits"] = self.tensor_parallel_word_embeddings

        return batch

    def _prepare_inputs_ids_and_labels_for_forward(self, batch: dict) -> tuple[torch.Tensor]:
        if self.is_pipeline_parallel_enabled:
            # when using pipeline parallel, we broadcast the input outside the model function
            tokens = batch["text"]
            tokens = tokens.to(torch.cuda.current_device())

            if self.pipeline_stage_id == 0:
                input_ids = tokens[:, :-1]
            else:
                input_ids = tokens

            labels = None
        else:
            if ProcessGroupManager.is_tensor_parallel_enabled():
                tokens = self.broadcast_tensor_parallel_input(
                    None if batch is None else batch["text"], (self.micro_batch_size, self.sequence_length + 1)
                )
            else:
                tokens = batch["text"]
                tokens = tokens.to(torch.cuda.current_device())

            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]

        return input_ids, labels

    def _setup_model(self) -> None:
        super()._setup_model()

        assert not self.is_encoder_decoder, "currently encoder_decoder models are not supported for pretraining"

        if self.use_padding_free_transformer:
            if not self.reset_attention_mask:
                self.register_buffer(
                    "cu_seqlens",
                    torch.arange(
                        0,
                        self.micro_batch_size * self.sequence_length + 1,
                        self.sequence_length,
                        dtype=torch.int32,
                        device=torch.cuda.current_device(),
                    ),
                    persistent=False,
                )
                self.max_seqlen = self.sequence_length
            if self.reset_position_ids:
                assert self.reset_attention_mask, "reset_attention_mask should be specified with reset_position_ids"
            else:
                self.register_buffer(
                    "position_ids",
                    torch.arange(0, self.sequence_length, 1, device=torch.cuda.current_device()).repeat(
                        self.micro_batch_size
                    ),
                    persistent=False,
                )
        else:
            assert (
                not self.reset_attention_mask
            ), "currently reset_attention_mask is only implemented for padding free transformer"
            assert (
                not self.reset_position_ids
            ), "currently reset_position_ids is only implemented for padding free transformer"
