import logging
from typing import List, Union

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import AttentionImplementation, DistributedBackend, GradientCheckpointingMethod, LossMask, Mode
from ..hf_models import get_tensor_parallel_class, is_custom_model, is_tensor_parallel_compatible_model
from ..hf_models.modeling_utils import is_glu
from ..utils import ProcessGroupManager, log_rank_0, string_to_torch_dtype


class ModelWrapper(torch.nn.Module):
    """Model class which wraps any HuggingFace model"""

    def __init__(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode):
        """initializes a Model wrapper for a HuggingFace model

        Args:
            args (Union[TrainingArgs, InferenceArgs, ExportArgs]): arguments based on training / inference mode
            mode (Mode): training / inference mode for running the program
        """

        super().__init__()

        self.mode = mode
        self.model_name = args.model_args.model_name
        self.model_class = args.model_args.model_class
        self.gradient_checkpointing_method = args.distributed_args.gradient_checkpointing_method
        self.gradient_checkpointing_args = args.distributed_args.gradient_checkpointing_args
        self.efficient_initialization = args.model_args.efficient_initialization
        self.tuning_method = args.tuning_args.tuning_method
        self.dtype = args.mixed_precision_args.dtype
        self.reset_attention_mask = args.model_args.reset_attention_mask
        self.reset_position_ids = args.model_args.reset_position_ids
        self.attention_implementation = args.model_args.attention_implementation
        self.use_padding_free_transformer = args.model_args.use_padding_free_transformer
        self.tensor_parallel_embeddings = args.distributed_args.tensor_parallel_embeddings

        self.tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.distributed_backend = None
        self.stage = None
        if self.mode == Mode.training:
            self.distributed_backend = args.distributed_args.distributed_backend
            self.stage = args.distributed_args.stage

        self._setup_config(args)
        self.tie_word_embeddings = self.config.tie_word_embeddings
        self.is_encoder_decoder = self.config.is_encoder_decoder

        if self.tp_world_size > 1:
            self.model_class = get_tensor_parallel_class(self.config.model_type)

            assert is_tensor_parallel_compatible_model(
                self.model_class, self.config.model_type
            ), "tensor parallel is not supported with this model"

        if self.use_padding_free_transformer:
            assert is_custom_model(
                self.model_class, self.config.model_type
            ), "padding free transformer is not supported with the specified model"

            assert (
                self.attention_implementation == AttentionImplementation.flash_attention_2
            ), "padding free transformer only works with flash attention"

            assert self.tp_world_size == 1, "padding free transformer is not supported with tensor parallel"

        self._setup_tokenizer(args)
        self._setup_model(args)

        self.loss_mask = None
        if self.mode == Mode.training:
            self.loss_mask = args.training_parameters.loss_mask
            if self.is_encoder_decoder:
                assert (
                    self.loss_mask == LossMask.output_only
                ), "only output_only loss mask is supported with encoder decoder models"

        if self.mode == Mode.training:
            neft_alpha = args.research_args.neft_alpha
            if neft_alpha is not None and neft_alpha > 0:
                self._override_embedding_forward_with_neft_forward(neft_alpha)

        additional_special_tokens = args.tokenizer_args.additional_special_tokens
        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            original_vocab_size = len(self.tokenizer)

            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            log_rank_0(logging.INFO, f"added {len(additional_special_tokens)} tokens")

            if len(self.tokenizer) != original_vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))

    def generate(self, batch: dict, generate_kwargs: dict) -> List[str]:
        """generate function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch
            generate_kwargs (dict): generate kwargs for the batch

        Returns:
            List[str]: list of generated text. input is trimmed from the generated text
        """

        if self.use_padding_free_transformer:
            raise NotImplementedError("padding free transformer doesn't support generation")

        for i in batch:
            batch[i] = batch[i].to(torch.cuda.current_device())

        generated = self.model.generate(**batch, **generate_kwargs, eos_token_id=self.eos_token_id)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        # add 1 since eos token to also count eos in generated tokens
        num_generated_tokens = ((generated != self.eos_token_id).sum(dim=-1) + 1).tolist()
        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text, num_generated_tokens

    def get_model_tflops(self, batch_size: int, sequence_length: int) -> None:
        if not is_custom_model(self.model_class, self.config.model_type):
            return 0

        b = batch_size
        s = sequence_length
        h = self.config.n_embd
        f = self.config.n_inner
        n = self.config.n_head
        k = self.config.num_key_value_heads
        l = self.config.n_layer
        v = self.config.vocab_size

        mlp_flops = 4 * b * s * h * f
        if is_glu(self.config.activation_function):
            mlp_flops += 2 * b * s * h * f

        attention_flops = 4 * b * s * h * (h * (1 + k / n) + s)

        forward_flops = attention_flops + mlp_flops

        if self.gradient_checkpointing_method == GradientCheckpointingMethod.block:
            num_layers_checkpointed = l // self.gradient_checkpointing_args.get("checkpoint_every", 1)
            fraction_of_layers_checkpointed = num_layers_checkpointed / l
            backward_flops = (2 + fraction_of_layers_checkpointed) * forward_flops
        else:
            backward_flops = 2 * forward_flops

        model_flops = l * (forward_flops + backward_flops)
        model_flops += 6 * b * s * h * v
        model_flops /= 10**12

        return model_flops

    def _override_embedding_forward_with_neft_forward(self, neft_alpha: float):
        if not hasattr(self.model, "get_input_embeddings"):
            raise Exception(
                "`get_input_embeddings` is not implemented for this model so its not possible to inject noise to input"
                " embeddings. Please implement `get_input_embeddings` ot set `neft_alpha` to None"
            )

        original_forward = self.model.get_input_embeddings().forward

        def _noisy_forward(x: torch.Tensor):
            x = original_forward(x)

            # to check if we are in eval mode we use self.training instead of self.model.training
            if self.training:
                mag_norm = neft_alpha / torch.sqrt(torch.tensor(torch.numel(x)))
                return x + torch.zeros_like(x).uniform_(-mag_norm, mag_norm)

            return x

        # overrides the forward function of torch.nn.Embedding
        self.model.get_input_embeddings().forward = _noisy_forward

    def save_pretrained(self, save_path: str) -> None:
        self.tokenizer.save_pretrained(save_path, legacy_format=False)
        self.model.save_pretrained(save_path)

    def _setup_config(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
        if self.model_name is None:
            self.config = AutoConfig.for_model(**args.model_args.pretrained_config)
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=args.model_args.trust_remote_code
            )
        log_rank_0(logging.INFO, self.config)

    def _setup_tokenizer(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
        tokenizer_name = args.tokenizer_args.tokenizer_name if self.model_name is None else self.model_name
        assert tokenizer_name is not None, "pass a tokenizer"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

    def _setup_model(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
        if self.model_name is None:
            model_kwargs = {"config": self.config}
        else:
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": args.model_args.trust_remote_code,
            }

        if self.attention_implementation is not None:
            model_kwargs["attn_implementation"] = self.attention_implementation.value
        if self.use_padding_free_transformer:
            model_kwargs["use_padding_free_transformer"] = True
        if self.tensor_parallel_embeddings:
            model_kwargs["tensor_parallel_embeddings"] = True

        def _get_model(**extras):
            if self.model_name is None:
                if self.tp_world_size > 1:
                    # avoid inferring the model class so use _from_config instead of from_config
                    model = self.model_class._from_config(**model_kwargs, **extras)
                else:
                    model = self.model_class.from_config(**model_kwargs, **extras)
            else:
                model = self.model_class.from_pretrained(**model_kwargs, **extras)

            return model

        if self.mode == Mode.training:
            if self.distributed_backend == DistributedBackend.deepspeed:
                if self.efficient_initialization:
                    from ..distributed import get_deepspeed_config

                    self.deepspeed_config = HfDeepSpeedConfig(get_deepspeed_config(args))

                self.model = _get_model()
            elif self.distributed_backend == DistributedBackend.torch:
                if self.efficient_initialization:
                    if self.model_name is None:
                        with torch.device("meta"):
                            self.model = _get_model()
                    else:
                        if is_custom_model(self.model_class, self.config.model_type):
                            if ProcessGroupManager.get_data_parallel_rank() == 0:
                                self.model = _get_model()
                            else:
                                with torch.device("meta"):
                                    self.model = _get_model()
                        else:
                            self.model = _get_model(low_cpu_mem_usage=True)
                else:
                    self.model = _get_model()
        else:
            if self.dtype == "fp8":
                log_rank_0(logging.WARN, "dtype fp8 was passed but loading model in fp16")
                torch_dtype = torch.float16
            else:
                torch_dtype = string_to_torch_dtype(self.dtype)

            self.model = _get_model(device_map=torch.cuda.current_device(), torch_dtype=torch_dtype)

        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.numel()

        log_rank_0(logging.INFO, f"num parameters in the model = {num_parameters:,}")
