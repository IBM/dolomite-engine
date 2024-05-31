from typing import Union

from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode, TuningMethod
from ..utils import string_to_torch_dtype
from .finetuning import ModelWrapperForFinetuning


class ModelWrapperForPEFT(ModelWrapperForFinetuning):
    def __init__(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode):
        super().__init__(args, mode)

        assert not self.reset_attention_mask, "reset_attention_mask is only supported with pretraining"
        assert not self.reset_position_ids, "reset_position_ids is only supported with pretraining"

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

        assert not self.use_padding_free_transformer

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

        self.model = args.model_args.model_class.from_pretrained(
            **model_kwargs, torch_dtype=string_to_torch_dtype(self.dtype)
        )
        self.model = get_peft_model(self.model, self.peft_config)
