import json
from argparse import ArgumentParser
from typing import Any, List, Union

import numpy as np
import torch
import transformers
from peft import PromptTuningInit
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from engine.constants import (
    AttentionImplementation,
    DatasetConfigKeys,
    LearningRateScheduler,
    Mode,
    OptimizerKeys,
    PaddingSide,
    TrainingInferenceType,
)


class BaseArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModelArgs(BaseArgs):
    # model name on huggingface hub
    model_name: str = None
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # dtype to use for training / inference
    dtype: str = "float32"
    # add special tokens to the tokenizer
    additional_special_tokens: List[str] = None
    # trust remote code for models that are not directly supported by HuggingFace yet
    trust_remote_code: bool = False
    # padding side
    padding_side: PaddingSide = None
    # attention implementation (only works with GPTMegatronForCausalLM)
    attention_implementation: AttentionImplementation = None

    def model_post_init(self, __context: Any) -> None:
        # model_name
        assert self.model_name is not None, "model_name cannot be None"

        # model_class
        if self.attention_implementation is None:
            assert self.model_class in [
                AutoModelForCausalLM.__name__,
                AutoModelForSeq2SeqLM.__name__,
            ], f"attention implementation is not supported with {AutoModelForCausalLM.__name__} or {AutoModelForSeq2SeqLM.__name__}"

            self.model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM] = getattr(
                transformers, self.model_class
            )
        else:
            try:
                from ibm_models import GPTMegatronForCausalLM

                self.model_class = GPTMegatronForCausalLM
            except ImportError:
                raise ImportError(
                    "please pip install ibm-models: https://github.ibm.com/ai-models-architectures/ibm-models"
                )

        # dtype
        self.dtype = getattr(torch, self.dtype)
        assert self.dtype in [torch.float32, torch.float16, torch.bfloat16], f"unexpected dtype '{self.dtype}'"


class InitializationArgs(BaseArgs):
    # random seed
    seed: int = 42
    # type of tuning, full finetuning or PEFT
    training_inference_type: TrainingInferenceType = None
    # prompt tuning init method
    prompt_tuning_init: PromptTuningInit = None
    # prompt tuning init text
    prompt_tuning_init_text: str = None
    # number of virtual tokens for PEFT
    num_virtual_tokens: int = None
    # the dimension of the low-rank matrices
    lora_rank: int = None
    # the scaling factor for the low-rank matrices
    lora_alpha: float = 32.0
    # the dropout probability of the LoRA layers
    lora_dropout: float = 0.1
    # path to load checkpoints
    load_path: str = None

    def model_post_init(self, __context: Any) -> None:
        assert self.training_inference_type is not None, "training_inference_type can't be None"

        # check whether the arguments specified are valid
        if self.training_inference_type == TrainingInferenceType.full_finetuning:
            self._check_prompt_tuning_is_disabled()
            self._check_lora_is_disabled()

        elif self.training_inference_type == TrainingInferenceType.prompt_tuning:
            if self.prompt_tuning_init == PromptTuningInit.RANDOM:
                assert (
                    self.prompt_tuning_init_text is None
                ), f"prompt_tuning_init_text '{self.prompt_tuning_init_text}' was specified with RANDOM init method"
            elif self.prompt_tuning_init == PromptTuningInit.TEXT:
                assert (
                    self.prompt_tuning_init_text is not None
                ), f"prompt_tuning_init_text needs to be specified with TEXT init method"

            self._check_lora_is_disabled()

        elif self.training_inference_type == TrainingInferenceType.lora:
            assert self.lora_rank is not None, f"lora_rank {self.lora_rank} is a required argument for lora"

            self._check_prompt_tuning_is_disabled()

    def _check_prompt_tuning_is_disabled(self) -> None:
        assert (
            self.prompt_tuning_init is None
        ), f"prompt_tuning_init '{self.prompt_tuning_init}' should not be specified with {self.training_inference_type.value}"
        assert (
            self.prompt_tuning_init_text is None
        ), f"prompt_tuning_init_text '{self.prompt_tuning_init_text}' should not be specified with {self.training_inference_type.value}"
        assert (
            self.num_virtual_tokens is None
        ), f"num_virtual_tokens '{self.num_virtual_tokens}' should not be specified with {self.training_inference_type.value}"

    def _check_lora_is_disabled(self) -> None:
        assert self.lora_rank is None, f"lora_rank {self.lora_rank} should not be specified with full_finetuning"


class DatasetArgs(BaseArgs):
    # list of datasets to use
    datasets: List[dict] = []

    def model_post_init(self, __context: Any) -> None:
        # datasets
        assert self.datasets is not None and len(self.datasets) != 0, "datasets cannot be None or an empty list"
        self._check_each_dataset_and_set_defaults()

    def _check_each_dataset_and_set_defaults(self) -> None:
        """checks whether the arguments specified in the config are valid"""

        import engine.data as data_classes

        for i, data_config in enumerate(self.datasets):
            assert (
                DatasetConfigKeys.data_class.value in data_config
            ), f"{DatasetConfigKeys.data_class.value} is not specified for dataset at index {i}"
            # convert to string to the actual class type
            data_config[DatasetConfigKeys.data_class.value] = getattr(
                data_classes, data_config[DatasetConfigKeys.data_class.value]
            )

            # check data_sampling_proportion
            assert (
                DatasetConfigKeys.data_sampling_proportion.value in data_config
                and isinstance(data_config[DatasetConfigKeys.data_sampling_proportion.value], int)
                and data_config[DatasetConfigKeys.data_sampling_proportion.value] > 0
            ), f"{DatasetConfigKeys.data_sampling_proportion.value} is not specified for dataset at index {i}"


class OptimizationArgs(BaseArgs):
    # optimizer
    optimizer: dict = {
        "optimizer_class": "ApexFusedAdam",
        "lr": 1e-5,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-10,
    }
    # learning rate schedule
    lr_schedule: LearningRateScheduler = LearningRateScheduler.cosine
    # warmup steps
    warmup_steps: int = 200

    def model_post_init(self, __context: Any) -> None:
        # optimizer
        import engine.optimization as optimizer_classes

        self.optimizer[OptimizerKeys.optimizer_class.value] = getattr(
            optimizer_classes, self.optimizer[OptimizerKeys.optimizer_class.value]
        )


class DeepSpeedArgs(BaseArgs):
    # deepspeed ZeRO stage
    stage: int = 3
    # overlap communication with computation
    overlap_comm: bool = False
    # use contiguous buffers for gradients, requires more memory if enabled
    contiguous_gradients: bool = False
    # train with CPU offloading to save GPU memory
    cpu_offload: bool = False


class LoggingArgs(BaseArgs):
    # logging directory for experiments
    logdir: str = None
    # aim repo, experiment logs are saved here
    aim_repo: str = None
    # name of the experiment
    experiment_name: str = None


class DebuggingArgs(BaseArgs):
    # steps per print for memory logging etc for deepspeed
    steps_per_print: int = np.inf


class TrainingArgs(
    ModelArgs, InitializationArgs, DatasetArgs, OptimizationArgs, DeepSpeedArgs, LoggingArgs, DebuggingArgs
):
    # path to save checkpoints
    save_path: str = None
    # whether to use sequential sampler for validation
    ignore_sampling_proportion_for_validation: bool = False
    # number of training steps
    num_training_steps: int = None
    # gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # interval for evaluation
    eval_interval: int = None
    # interval for checkpointing
    save_interval: int = None
    # batch size per GPU for ZeRO-DP
    batch_size_per_gpu: int = None
    # whether to use val dataset for validation during training
    eval_during_training: bool = True
    # whether to use gradient checkpointing, enabling leads to lower memory usage with increased step time
    gradient_checkpointing: bool = False

    def model_post_init(self, __context: Any) -> None:
        ModelArgs.model_post_init(self, __context)
        InitializationArgs.model_post_init(self, __context)
        DatasetArgs.model_post_init(self, __context)
        OptimizationArgs.model_post_init(self, __context)
        DeepSpeedArgs.model_post_init(self, __context)
        LoggingArgs.model_post_init(self, __context)
        DebuggingArgs.model_post_init(self, __context)

        # save_path
        assert self.save_path is not None, "save_path cannot be None"

        # num_training_steps
        assert self.num_training_steps is not None, "num_training_steps cannot be None"

        # save_interval
        assert self.save_interval is not None, "save_interval cannot be None"

        # eval_interval
        if self.eval_during_training:
            assert self.eval_interval is not None, "eval_interval cannot be None"

        # batch_size_per_gpu
        assert self.batch_size_per_gpu is not None, "batch_size_per_gpu cannot be None"


class InferenceArgs(ModelArgs, InitializationArgs, DatasetArgs):
    # batch size
    batch_size: int = None
    # sample or greedy
    do_sample: bool = None
    # max new tokens to generate
    max_new_tokens: int = None
    # temperature
    temperature: float = None
    # top k
    top_k: int = None
    # top p
    top_p: float = None
    # output dir
    output_dir: str = None

    def model_post_init(self, __context: Any) -> None:
        ModelArgs.model_post_init(self, __context)
        InitializationArgs.model_post_init(self, __context)
        DatasetArgs.model_post_init(self, __context)

        # load_path
        if self.load_path is None:
            from engine.utils.logging import warn_rank_0

            warn_rank_0("load_path was None, not loading any trained checkpoint")

        # batch_size
        assert self.batch_size is not None, "batch_size cannot be None"

        # max_new_tokens
        assert self.max_new_tokens is not None, "max_new_tokens cannot be None"

        # output_dir
        assert self.output_dir is not None, "output_dir cannot be None"


def get_args(mode: Mode) -> Union[TrainingArgs, InferenceArgs]:
    """get args for training / inference

    Args:
        mode (Mode): training / inference mode for running the program

    Returns:
        Union[TrainingArgs, InferenceArgs]: args for training / inference
    """

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path for the config")
    args = parser.parse_args()

    config: dict = json.load(open(args.config, "r"))

    if mode == Mode.training:
        args = TrainingArgs(**config)
    else:
        args = InferenceArgs(**config)

    from engine.utils import print_args

    print_args(args)
    return args
