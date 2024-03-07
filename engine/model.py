import logging
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from .arguments import ExportArgs, InferenceArgs, TrainingArgs
from .distributed import get_deepspeed_config
from .enums import AttentionImplementation, DistributedBackend, Mode, PaddingSide, TuningMethod
from .utils import get_local_rank, log_rank_0, register_profiler, register_timer, run_rank_n, warn_rank_0


class Model(torch.nn.Module):
    """Model class which wraps any HuggingFace model"""

    @register_profiler("initialize_model")
    @register_timer("initialize_model")
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

        self._setup_input_device()

        self.attention_implementation = args.model_args.attention_implementation
        if self.attention_implementation is not None:
            from ibm_models import GPTMegatronForCausalLM, MoEMegablocksForCausalLM

            assert self.model_class in [GPTMegatronForCausalLM, MoEMegablocksForCausalLM]

        self.distributed_backend = args.distributed_args.distributed_backend if mode == Mode.training else None

        self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=args.model_args.trust_remote_code)

        self.is_encoder_decoder = self.config.is_encoder_decoder
        self.tuning_method = args.tuning_args.tuning_method
        self.dtype = args.model_args.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.padding_side = PaddingSide(
            self.tokenizer.padding_side
            if args.tokenizer_args.padding_side is None
            else args.tokenizer_args.padding_side
        )

        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": args.model_args.trust_remote_code,
            "use_cache": mode == Mode.inference,
        }
        if self.attention_implementation is not None:
            model_kwargs["attention_implementation"] = self.attention_implementation

        if self.tuning_method == TuningMethod.pretraining:
            self._setup_model_for_pretraining(args, model_kwargs)
        elif self.tuning_method == TuningMethod.full_finetuning:
            self._setup_model_for_finetuning(args, model_kwargs)
        elif self.tuning_method in [TuningMethod.prompt_tuning, TuningMethod.lora]:
            self._setup_model_for_peft(args, model_kwargs)
        else:
            raise ValueError(f"unexpected tuning_method ({self.tuning_method})")

        neft_alpha = args.research_args.neft_alpha
        if neft_alpha is not None and neft_alpha > 0:
            self._override_embedding_forward_with_neft_forward(neft_alpha)

        if args.tokenizer_args.additional_special_tokens is not None:
            original_vocab_size = len(self.tokenizer)

            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": args.tokenizer_args.additional_special_tokens}
            )
            log_rank_0(logging.INFO, f"added {len(args.tokenizer_args.additional_special_tokens)} tokens")

            if len(self.tokenizer) != original_vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))

    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: Tuple[List[int]]) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        if self.tuning_method == TuningMethod.pretraining:
            # for pretraining we compute loss externally here instead of relying on transformers.
            # this is done because megatron's dataset returns batches of length (sequence_length + 1)
            # instead of (sequence_length), so we need to trim the input_ids before forward pass.
            # transformers does forward pass before however and then trims the tokens.

            tokens: torch.Tensor = batch["text"]
            if not tokens.is_cuda:
                tokens = tokens.to(self.input_device)

            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]

            if self.attention_implementation == AttentionImplementation.packed_flash:
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
        else:
            batch = self.prepare_batch(batch)

            if self.attention_implementation != AttentionImplementation.packed_flash:
                for i in batch:
                    batch[i] = batch[i].to(self.input_device)

            model_outputs = self.model(**batch)

            if type(model_outputs) is tuple:
                loss = model_outputs[0]
            else:
                loss = model_outputs.loss

        return loss

    @register_profiler("generate")
    @register_timer("generate")
    def generate(self, batch: Tuple[List[int]], generate_kwargs: dict) -> List[str]:
        """generate function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch
            generate_kwargs (dict): generate kwargs for the batch

        Returns:
            List[str]: list of generated text. input is trimmed from the generated text
        """

        if self.attention_implementation == AttentionImplementation.packed_flash:
            raise NotImplementedError("packed_flash attention doesn't support generation yet")

        batch = self.prepare_batch(batch)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        generated = self.model.generate(**batch, **generate_kwargs, eos_token_id=self.tokenizer.eos_token_id)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        # add 1 since eos token to also count eos in generated tokens
        num_generated_tokens = ((generated != self.tokenizer.eos_token_id).sum(dim=-1) + 1).tolist()
        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text, num_generated_tokens

    @register_profiler("prepare_batch")
    @register_timer("prepare_batch")
    def prepare_batch(self, batch: Tuple[List[int]]) -> dict:
        """prepares the batch with padding to pass into the forward function of the HuggingFace model

        Args:
            batch (Tuple[List[int]]): input tokens and output tokens. Output tokens are optional when running generation but required for training.

        Returns:
            dict: dict containing input_ids, attention_mask and labels if outputs is specified
        """

        result = {}

        if self.mode == Mode.training:
            inputs, outputs = batch
            assert outputs is not None, "outputs can't be None during training"
        else:
            inputs = batch
            outputs = None

        input_ids, attention_mask, labels = _pad(
            inputs,
            outputs,
            self.tokenizer.eos_token_id,
            padding_side=self.padding_side,
            is_encoder_decoder=self.is_encoder_decoder,
            attention_implementation=self.attention_implementation,
        )

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
        if self.mode == Mode.training:
            result["labels"] = labels

        return result

    def _setup_model_for_pretraining(
        self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], model_kwargs: dict
    ) -> None:
        # cache these since they are static during pretraining
        if self.attention_implementation == AttentionImplementation.packed_flash:
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

        self._setup_model_for_finetuning(args, model_kwargs)

    def _setup_model_for_finetuning(
        self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], model_kwargs: dict
    ) -> None:
        if self.mode == Mode.training:
            # this tells from_pretrained to instantiate directly on gpus
            # this only instantiates a single instance of the model across the ranks
            if self.distributed_backend == DistributedBackend.deepspeed:
                self.deepspeed_config = HfDeepSpeedConfig(get_deepspeed_config(args))

            self.model = args.model_args.model_class.from_pretrained(**model_kwargs)

            if args.distributed_args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model = args.model_args.model_class.from_pretrained(**model_kwargs, torch_dtype=self.dtype)

    def _setup_model_for_peft(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], model_kwargs: dict) -> None:
        if args.tuning_args.tuning_method == TuningMethod.prompt_tuning:
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                prompt_tuning_init=args.tuning_args.prompt_tuning_args.prompt_tuning_init,
                num_virtual_tokens=args.tuning_args.prompt_tuning_args.num_virtual_tokens,
                prompt_tuning_init_text=args.tuning_args.prompt_tuning_args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.model_args.model_name,
            )
        elif args.tuning_args.tuning_method == TuningMethod.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                inference_mode=self.mode != Mode.training,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

        self.model = args.model_args.model_class.from_pretrained(**model_kwargs, torch_dtype=self.dtype)

        if args.distributed_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model = get_peft_model(self.model, self.peft_config)

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

    def _setup_input_device(self) -> None:
        if self.mode == Mode.training:
            # if using deepspeed
            self.input_device = get_local_rank()
        else:
            self.input_device = 0
            if not torch.cuda.is_available():
                warn_rank_0("no CUDA device found, running on CPU")
                self.input_device = "cpu"


def _pad(
    inputs: list,
    outputs: list,
    pad_token_id: int,
    padding_side: PaddingSide,
    is_encoder_decoder: bool,
    attention_implementation: AttentionImplementation = None,
) -> Tuple[List[int], List[int]]:
    """pads the arrays with the specified padding value

    Args:
        inputs (list): input token ids
        outputs (list): output token labels
        pad_token_id (int): token id to pad with
        padding_side (PaddingSide): padding side for the tensors
        is_encoder_decoder (bool): whether the model is an encoder-decoder or a decoder-only model
        attention_implementation (AttentionImplementation): attention implementation for the model

    Returns:
        Tuple[List[int], List[int]]: token ids and the corresponding attention masks
    """

    # labels is None when outputs is None
    labels = None

    if is_encoder_decoder:
        input_max_length = max(list(map(len, inputs)))

        if padding_side == PaddingSide.left:
            input_ids = [[pad_token_id] * (input_max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (input_max_length - len(array)) + [1] * len(array) for array in inputs]
        else:
            input_ids = [array + [pad_token_id] * (input_max_length - len(array)) for array in inputs]
            attention_mask = [[1] * len(array) + [0] * (input_max_length - len(array)) for array in inputs]

        if outputs is not None:
            output_max_length = max(list(map(len, outputs)))
            labels = [array + [-100] * (output_max_length - len(array)) for array in outputs]
    else:
        if attention_implementation == AttentionImplementation.packed_flash:
            input_ids = inputs
            attention_mask = None
            labels = [
                [-100] * (len(array_in) - len(array_out)) + array_out for array_in, array_out in zip(inputs, outputs)
            ]
        else:
            max_length = max(list(map(len, inputs)))

            if padding_side == PaddingSide.left:
                input_ids = [[pad_token_id] * (max_length - len(array)) + array for array in inputs]
                attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in inputs]

                if outputs is not None:
                    labels = [[-100] * (max_length - len(array)) + array for array in outputs]
            else:
                input_ids = [array + [pad_token_id] * (max_length - len(array)) for array in inputs]
                attention_mask = [[1] * len(array) + [0] * (max_length - len(array)) for array in inputs]

                if outputs is not None:
                    labels = [
                        [-100] * (len(array_in) - len(array_out)) + array_out + [-100] * (max_length - len(array_in))
                        for array_in, array_out in zip(inputs, outputs)
                    ]

    if attention_implementation != AttentionImplementation.packed_flash:
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        if labels is not None:
            labels = torch.tensor(labels)

    return input_ids, attention_mask, labels


@run_rank_n
def log_model(model: Model) -> None:
    """print model

    Args:
        model (Model): model to print
    """

    log_rank_0(logging.INFO, "------------------------ model ------------------------")
    log_rank_0(logging.INFO, model)
    log_rank_0(logging.INFO, "-------------------- end of model ---------------------")
