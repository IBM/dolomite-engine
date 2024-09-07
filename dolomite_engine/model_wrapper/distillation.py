import logging

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..enums import AttentionImplementation, DistributedBackend, Mode
from ..hf_models.modeling_utils_TP import tensor_to_dtensor
from ..utils import ProcessGroupManager, log_rank_0, string_to_torch_dtype
from .pretraining import ModelWrapperForPretraining


class ModelWrapperForDistillation(ModelWrapperForPretraining):
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
        micro_batch_size: int,
        sequence_length: int,
        teacher_model_name: str | None,
        teacher_model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        teacher_model_dtype: torch.dtype,
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
            distributed_backend (DistributedBackend): distributed backend to use for model
            micro_batch_size (int): micro batch size for pretraining
            sequence_length (int): sequence length for pretraining
            neft_alpha (float | None, optional): alpha parameter for NEFTune. Defaults to None.
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            reset_attention_mask (bool, optional): whether to reset attention mask during pretraining. Defaults to False.
            reset_position_ids (bool, optional): whether to reset position ids during pretraining. Defaults to False.
        """

        self.teacher_model_class = teacher_model_class
        self.teacher_model_name = teacher_model_name
        self.teacher_model_dtype = teacher_model_dtype

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
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
            neft_alpha=neft_alpha,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
            reset_attention_mask=reset_attention_mask,
            reset_position_ids=reset_position_ids,
        )

        if self.tp_world_size > 1:
            raise NotImplementedError()

    def _setup_config(self) -> None:
        super()._setup_config()

        self.teacher_config = AutoConfig.from_pretrained(
            self.teacher_model_name, trust_remote_code=self.trust_remote_code
        )

    def _setup_tokenizer(self) -> None:
        super()._setup_tokenizer()

        log_rank_0(
            logging.WARN,
            "tokenizers should be same for both teacher and student, unfortunately I don't know how to check for this",
        )

    def _setup_model(self) -> None:
        super()._setup_model()

        self.teacher_model = self.teacher_model_class.from_pretrained(
            self.teacher_model_name, torch_dtype=string_to_torch_dtype(self.teacher_model_dtype)
        )
