from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..enums import AttentionImplementation, DistributedBackend, Mode
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
        use_padding_free_transformer: bool,
        tensor_parallel_word_embeddings: bool,
        sequence_parallel: bool,
        distributed_backend: DistributedBackend,
        random_seed: int,
        micro_batch_size: int,
        sequence_length: int,
        neft_alpha: float | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        upcast_logits_for_loss: bool = False,
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
            distributed_backend (DistributedBackend): distributed backend to use for model
            random_seed (int): random seed to use for tensor parallel seed management
            micro_batch_size (int): micro batch size for pretraining
            sequence_length (int): sequence length for pretraining
            neft_alpha (float | None, optional): alpha parameter for NEFTune. Defaults to None.
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            upcast_logits_for_loss (bool, optional): whether to upcast logits for loss computation
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
            use_padding_free_transformer=use_padding_free_transformer,
            tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            sequence_parallel=sequence_parallel,
            distributed_backend=distributed_backend,
            random_seed=random_seed,
            neft_alpha=neft_alpha,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
            upcast_logits_for_loss=upcast_logits_for_loss,
        )

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

        input_ids, labels = self._prepare_inputs_ids_and_labels_for_forward(batch)
        batch = self._prepare_model_inputs(input_ids)

        model_outputs = self.model(**batch)
        logits: torch.Tensor = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.logits

        if self.upcast_logits_for_loss:
            logits = logits.float()

        loss_context = nullcontext

        if self.tp_world_size > 1:
            logits = tensor_to_dtensor(
                logits, current_placement=Shard(-1) if self.tensor_parallel_word_embeddings else Replicate()
            )
            labels = tensor_to_dtensor(labels, current_placement=Replicate())

            if self.tensor_parallel_word_embeddings:
                loss_context = loss_parallel

        with loss_context():
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

        return loss

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

            batch["cu_seqlens"] = cu_seqlens
            batch["max_seqlen"] = max_seqlen
            batch["position_ids"] = position_ids

        batch["input_ids"] = input_ids

        if self.tp_world_size > 1:
            batch["output_parallel_lm_logits"] = self.tensor_parallel_word_embeddings

        return batch

    def _prepare_inputs_ids_and_labels_for_forward(self, batch: dict) -> torch.Tensor:
        if self.tp_world_size > 1:
            tp_source_rank = ProcessGroupManager.get_tensor_parallel_first_rank()
            tp_group = ProcessGroupManager.get_tensor_parallel_group()

            if self.tp_rank == 0:
                tokens: torch.Tensor = batch["text"]
                tokens = tokens.to(torch.cuda.current_device())
            else:
                tokens = torch.empty(
                    (self.micro_batch_size, self.sequence_length + 1),
                    dtype=torch.long,
                    device=torch.cuda.current_device(),
                )

            torch.distributed.broadcast(tokens, src=tp_source_rank, group=tp_group)
        else:
            tokens: torch.Tensor = batch["text"]
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
                self.register_buffer(
                    "max_seqlen",
                    torch.tensor(self.sequence_length, device=torch.cuda.current_device()),
                    persistent=False,
                )

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
