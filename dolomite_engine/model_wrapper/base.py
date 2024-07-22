import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from ..enums import AttentionImplementation, DistributedBackend, Mode
from ..hf_models import get_tensor_parallel_class, is_custom_model, is_tensor_parallel_compatible_model
from ..utils import (
    CUDA_RNGStatesTracker,
    ProcessGroupManager,
    SafeTensorsWeightsManager,
    log_rank_0,
    set_cuda_rng_tracker,
    string_to_torch_dtype,
)


class ModelWrapper(nn.Module):
    """Model class which wraps any HuggingFace model"""

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
        neft_alpha: float | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        upcast_logits_for_loss: bool = False,
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
            neft_alpha (float | None, optional): alpha parameter for NEFTune. Defaults to None.
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            upcast_logits_for_loss (bool, optional): whether to upcast logits for loss computation
        """

        super().__init__()

        self.mode = mode
        self.model_name = model_name
        self.pretrained_config = pretrained_config
        self.model_class = model_class
        self.efficient_initialization = efficient_initialization
        self.dtype = dtype
        self.attention_implementation = attention_implementation
        self.use_padding_free_transformer = use_padding_free_transformer
        self.tensor_parallel_word_embeddings = tensor_parallel_word_embeddings
        self.sequence_parallel = sequence_parallel
        self.tokenizer_name = tokenizer_name if self.model_name is None else self.model_name
        self.trust_remote_code = trust_remote_code
        self.upcast_logits_for_loss = upcast_logits_for_loss

        self.tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.distributed_backend = distributed_backend if self.mode == Mode.training else None

        self._setup_config()

        if self.tp_world_size > 1:
            self.model_class = get_tensor_parallel_class(self.config.model_type)

            assert is_tensor_parallel_compatible_model(
                self.model_class, self.config.model_type
            ), "tensor parallel is not supported with this model"

            rng_tracker = CUDA_RNGStatesTracker()
            rng_tracker.add(seed=random_seed)
            set_cuda_rng_tracker(rng_tracker)

        if self.use_padding_free_transformer:
            assert is_custom_model(
                self.model_class, self.config.model_type
            ), "padding free transformer is not supported with the specified model"

            assert (
                self.attention_implementation == AttentionImplementation.flash_attention_2
            ), "padding free transformer only works with flash attention"

        self._setup_tokenizer()
        self._setup_model()

        if self.mode == Mode.training:
            if neft_alpha is not None and neft_alpha > 0:
                self._override_embedding_forward_with_neft_forward(neft_alpha)

        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            original_vocab_size = len(self.tokenizer)

            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            log_rank_0(logging.INFO, f"added {len(additional_special_tokens)} tokens")

            if len(self.tokenizer) != original_vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))

    def generate(self, batch: dict, generate_kwargs: dict) -> list[str]:
        """generate function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch
            generate_kwargs (dict): generate kwargs for the batch

        Returns:
            List[str]: list of generated text. input is trimmed from the generated text
        """

        if self.use_padding_free_transformer or self.tp_world_size > 1:
            raise NotImplementedError("padding free transformer and tensor parallel doesn't support generation")

        for i in batch:
            batch[i] = batch[i].to(torch.cuda.current_device())

        generated = self.model.generate(**batch, **generate_kwargs, eos_token_id=self.eos_token_id)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        # add 1 since eos token to also count eos in generated tokens
        num_generated_tokens = ((generated != self.eos_token_id).sum(dim=-1) + 1).tolist()
        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text, num_generated_tokens

    def save_pretrained(self, save_path: str, state_dict: dict | None = None) -> None:
        self.tokenizer.save_pretrained(save_path, legacy_format=False)

        if state_dict is None:
            self.model.save_pretrained(save_path)
        else:
            for key in list(state_dict.keys()):
                assert key.startswith("model.")
                state_dict[key.lstrip("model.")] = state_dict.pop(key)

            self.config.save_pretrained(save_path)
            SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

    def _setup_config(self) -> None:
        self.config = (
            AutoConfig.for_model(**self.pretrained_config)
            if self.model_name is None
            else AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        )

        self.tie_word_embeddings = self.config.tie_word_embeddings
        self.is_encoder_decoder = self.config.is_encoder_decoder

        log_rank_0(logging.INFO, self.config)

    def _setup_tokenizer(self) -> None:
        assert self.tokenizer_name is not None, "pass a tokenizer"

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

    def _setup_model(self) -> None:
        if self.model_name is None:
            model_kwargs = {"config": self.config}
        else:
            model_kwargs = {"pretrained_model_name_or_path": self.model_name}

        if self.attention_implementation is not None:
            model_kwargs["attn_implementation"] = self.attention_implementation.value
        if self.use_padding_free_transformer:
            model_kwargs["use_padding_free_transformer"] = True
        if self.tensor_parallel_word_embeddings:
            model_kwargs["tensor_parallel_word_embeddings"] = True
        if self.sequence_parallel:
            model_kwargs["sequence_parallel"] = True
        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        def _get_model(**extras):
            if self.model_name is None:
                assert self.upcast_logits_for_loss == getattr(model_kwargs["config"], "upcast_logits_for_loss", False)

                if self.tp_world_size > 1:
                    # avoid inferring the model class so use _from_config instead of from_config
                    model = self.model_class._from_config(**model_kwargs, **extras)
                else:
                    model = self.model_class.from_config(**model_kwargs, **extras)
            else:
                if self.upcast_logits_for_loss:
                    extras["upcast_logits_for_loss"] = True

                model = self.model_class.from_pretrained(**model_kwargs, **extras)

            return model

        if self.mode == Mode.training:
            if self.distributed_backend == DistributedBackend.deepspeed:
                if self.efficient_initialization:
                    from ..distributed import get_deepspeed_config

                    self.deepspeed_config = HfDeepSpeedConfig(get_deepspeed_config())

                self.model = _get_model()
            elif self.distributed_backend == DistributedBackend.torch:
                if self.efficient_initialization:
                    if self.model_name is None:
                        with torch.device("meta"):
                            self.model = _get_model()
                    else:
                        assert (
                            self.tp_world_size == 1
                        ), "tensor parallel models don't support efficient init with model name"

                        self.model = _get_model(low_cpu_mem_usage=True)
                else:
                    self.model = _get_model()
        else:
            if self.dtype == "fp8":
                log_rank_0(logging.WARN, "dtype fp8 was passed but loading model in fp16")
                torch_dtype = torch.float16
            else:
                torch_dtype = string_to_torch_dtype(self.dtype)

            self.model = _get_model(torch_dtype=torch_dtype)

        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.numel()

        log_rank_0(logging.INFO, f"num parameters in the model = {num_parameters:,}")

    def _override_embedding_forward_with_neft_forward(self, neft_alpha: float) -> None:
        if not hasattr(self.model, "get_input_embeddings"):
            raise Exception(
                "`get_input_embeddings` is not implemented for this model so its not possible to inject noise to input"
                " embeddings. Please implement `get_input_embeddings` ot set `neft_alpha` to None"
            )

        original_forward = self.model.get_input_embeddings().forward

        def _noisy_forward(x: torch.Tensor) -> torch.Tensor:
            x = original_forward(x)

            # to check if we are in eval mode we use self.training instead of self.model.training
            if self.training:
                mag_norm = neft_alpha / torch.sqrt(torch.tensor(torch.numel(x)))
                return x + torch.zeros_like(x).uniform_(-mag_norm, mag_norm)

            return x

        # overrides the forward function of torch.nn.Embedding
        self.model.get_input_embeddings().forward = _noisy_forward
