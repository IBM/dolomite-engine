from typing import Union

from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode, TuningMethod
from .finetuning import ModelForFinetuning


class ModelForPEFT(ModelForFinetuning):
    def _setup_model(self, args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
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
