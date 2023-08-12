from typing import List, Tuple, Union

import torch
from fm_nlp.architecture import GraniteHF, GraniteHFConfig, SandstoneHF, SandstoneHFConfig
from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import AttentionImplementation, Mode, PaddingSide, TrainingInferenceType
from src.utils import get_deepspeed_config, get_local_rank, register_profiler, register_timer, warn_rank_0
from src.utils.logging import print_rank_0


def pad(
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
        if attention_implementation == AttentionImplementation.flash:
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

    if attention_implementation != AttentionImplementation.flash:
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        if labels is not None:
            labels = torch.tensor(labels)

    return input_ids, attention_mask, labels


class Model(torch.nn.Module):
    """Model class which wraps any HuggingFace model"""

    @register_profiler("initialize_model")
    @register_timer("initialize_model")
    def __init__(self, args: Union[TrainingArgs, InferenceArgs], mode: Mode):
        """initializes a Model wrapper for a HuggingFace model

        Args:
            args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
            mode (Mode): training / inference mode for running the program
        """

        super().__init__()

        self.mode = mode
        self.model_name = args.model_name
        self.model_class = args.model_class
        self.attention_implementation = args.attention_implementation

        # check if model_class is GraniteHF
        if self.model_class == GraniteHF:
            self.config = GraniteHFConfig.from_pretrained(self.model_name)
        elif self.model_class == SandstoneHF:
            self.config = SandstoneHFConfig.from_pretrained(self.model_name)
        else:
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=args.trust_remote_code)

        self.is_encoder_decoder = self.config.is_encoder_decoder
        self.training_inference_type = args.training_inference_type
        self.dtype = args.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.original_vocab_size = len(self.tokenizer)

        self.padding_side = PaddingSide(
            self.tokenizer.padding_side if args.padding_side is None else args.padding_side
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print_rank_0(f"PAD token not found, adding it explicitely")

        if args.additional_special_tokens is not None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": args.additional_special_tokens})
            print_rank_0(f"added {len(args.additional_special_tokens)} tokens")

        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": args.trust_remote_code,
            "use_cache": mode == Mode.inference,
        }

        if self.training_inference_type == TrainingInferenceType.full_finetuning:
            self._setup_model_for_finetuning(args, model_kwargs)
        else:
            self._setup_model_for_peft(args, model_kwargs)

        print_rank_0(self.model)

        self._setup_input_device()

    def _setup_model_for_finetuning(self, args: Union[TrainingArgs, InferenceArgs], model_kwargs: dict) -> None:
        if self.mode == Mode.training:
            # this tells from_pretrained to instantiate directly on gpus
            # this only instantiates a single instance of the model across the ranks
            if self.model_class not in [GraniteHF, SandstoneHF]:
                self.deepspeed_config = HfDeepSpeedConfig(get_deepspeed_config(args))

            self.model = args.model_class.from_pretrained(**model_kwargs)

            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model = args.model_class.from_pretrained(**model_kwargs, torch_dtype=self.dtype)

        if self.attention_implementation is not None:
            self._inject_attention_implementation()

    def _setup_model_for_peft(self, args: Union[TrainingArgs, InferenceArgs], model_kwargs: dict) -> None:
        if args.training_inference_type == TrainingInferenceType.prompt_tuning:
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                prompt_tuning_init=args.prompt_tuning_init,
                num_virtual_tokens=args.num_virtual_tokens,
                prompt_tuning_init_text=args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.model_name,
            )
        elif args.training_inference_type == TrainingInferenceType.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                inference_mode=True if self.mode == Mode.inference else False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

        self.model = args.model_class.from_pretrained(**model_kwargs, torch_dtype=self.dtype)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.attention_implementation is not None:
            self._inject_attention_implementation()

        self.model = get_peft_model(self.model, self.peft_config)

    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: Tuple[List[int]]) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        batch = self.prepare_batch(batch)

        if self.attention_implementation != AttentionImplementation.flash:
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

        if self.attention_implementation == AttentionImplementation.flash:
            raise NotImplementedError("flash attention doesn't support generation yet")

        batch = self.prepare_batch(batch)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        generated = self.model.generate(**batch, **generate_kwargs, eos_token_id=self.tokenizer.eos_token_id)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        num_generated_tokens = (generated != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
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

        if self.mode == Mode.training:
            inputs, outputs = batch
        else:
            inputs = batch

        result = {}
        if self.mode == Mode.training:
            assert outputs is not None, "outputs can't be None during training"

            input_ids, attention_mask, labels = pad(
                inputs,
                outputs,
                self.tokenizer.pad_token_id,
                padding_side=self.padding_side,
                is_encoder_decoder=self.is_encoder_decoder,
                attention_implementation=self.attention_implementation,
            )

            result["labels"] = labels
        elif self.mode == Mode.inference:
            input_ids, attention_mask, _ = pad(
                inputs,
                None,
                self.tokenizer.pad_token_id,
                padding_side=self.padding_side,
                is_encoder_decoder=self.is_encoder_decoder,
                attention_implementation=self.attention_implementation,
            )

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        return result

    def _setup_input_device(self) -> None:
        if self.mode == Mode.training:
            # if using deepspeed
            self.input_device = get_local_rank()
        else:
            self.input_device = 0
            if not torch.cuda.is_available():
                warn_rank_0("no CUDA device found, running on CPU")
                self.input_device = "cpu"

            self.to(self.input_device)

    def post_init(self) -> None:
        """a post init method for expanding word embeddings"""

        if len(self.tokenizer) != self.original_vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _inject_attention_implementation(self) -> None:
        from transformers import GPTMegatronForCausalLM

        assert isinstance(self.model, GPTMegatronForCausalLM)

        if self.attention_implementation == AttentionImplementation.math:
            warn_rank_0("ignores padding and doesn't work for generation")
            self.model.inject_math_attention()
        elif self.attention_implementation == AttentionImplementation.flash:
            self.model.inject_flash_attention()
        elif self.attention_implementation == AttentionImplementation.sdpa:
            self.model.inject_sdpa()
        else:
            raise ValueError("unexpected attention_implementation")
