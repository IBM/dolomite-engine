import json
from argparse import ArgumentParser
from typing import Any, List, Union

import torch
import transformers
from peft import PromptTuningInit
from pydantic import BaseModel, Extra
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from src.constants import (
    DatasetConfigKeys,
    LearningRateScheduler,
    Mode,
    OptimizerKeys,
    PaddingSide,
    TrainingInferenceType,
)


class BaseArgs(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__.__config__.extra = Extra.allow
        __pydantic_self__._post_init()

    def _post_init(self) -> None:
        return


class ModelArgs(BaseArgs):
    # model name on huggingface hub
    model_name: str = None
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # dtype to use for training / inference
    dtype: str = "float32"
    # padding side
    padding_side: PaddingSide = None

    def _post_init(self) -> None:
        # model_name
        assert self.model_name is not None, "model_name cannot be None"

        # model_class
        self.model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM] = getattr(transformers, self.model_class)
        assert self.model_class in [
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        ], f"unexpected model_class '{self.model_class}'"

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
    # path to load checkpoints
    load_path: str = None

    def _post_init(self) -> None:
        # check whether the arguments specified are valid for finetuning / prompt tuning
        if self.training_inference_type == TrainingInferenceType.full_finetuning:
            assert (
                self.prompt_tuning_init is None
            ), f"prompt_tuning_init '{self.prompt_tuning_init}' should not be specified with full_finetuning"
            assert (
                self.prompt_tuning_init_text is None
            ), f"prompt_tuning_init_text '{self.prompt_tuning_init_text}' should not be specified with full_finetuning"
            assert (
                self.num_virtual_tokens is None
            ), f"num_virtual_tokens '{self.num_virtual_tokens}' should not be specified with full_finetuning"
        elif self.training_inference_type == TrainingInferenceType.prompt_tuning:
            if self.prompt_tuning_init == PromptTuningInit.RANDOM:
                assert (
                    self.prompt_tuning_init_text is None
                ), f"prompt_tuning_init_text '{self.prompt_tuning_init_text}' was specified with RANDOM init method"
            elif self.prompt_tuning_init == PromptTuningInit.TEXT:
                assert (
                    self.prompt_tuning_init_text is not None
                ), f"prompt_tuning_init_text needs to be specified with TEXT init method"


class DatasetArgs(BaseArgs):
    # list of datasets to use
    datasets: List[dict] = []

    def _post_init(self) -> None:
        # datasets
        assert self.datasets is not None and len(self.datasets) != 0, "datasets cannot be None or an empty list"
        self._check_each_dataset_and_set_defaults()

    def _check_each_dataset_and_set_defaults(self) -> None:
        """checks whether the arguments specified in the config are valid"""

        import src.data as data_classes

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

    def _post_init(self) -> None:
        # optimizer
        import src.optimization as optimizer_classes

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
    # steps per print
    steps_per_print: int = 10


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
    # interval for evaluation and checkpointing
    eval_and_save_interval: int = None
    # batch size per GPU for ZeRO-DP
    batch_size_per_gpu: int = None
    # whether to use val dataset for validation during training
    eval_during_training: bool = True

    def _post_init(self) -> None:
        ModelArgs._post_init(self)
        InitializationArgs._post_init(self)
        DatasetArgs._post_init(self)
        OptimizationArgs._post_init(self)
        DeepSpeedArgs._post_init(self)
        LoggingArgs._post_init(self)
        DebuggingArgs._post_init(self)

        # save_path
        assert self.save_path is not None, "save_path cannot be None"

        # num_training_steps
        assert self.num_training_steps is not None, "num_training_steps cannot be None"

        # eval_and_save_interval
        assert self.eval_and_save_interval is not None, "eval_and_save_interval cannot be None"

        # batch_size_per_gpu
        assert self.batch_size_per_gpu is not None, "batch_size_per_gpu cannot be None"


class InferenceArgs(ModelArgs, InitializationArgs, DatasetArgs):
    # batch size
    batch_size: int = None
    # sample or greedy
    do_sample: bool = False
    # max new tokens to generate
    max_new_tokens: int = None
    # temperature
    temperature: float = None
    # top k
    top_k: int = None
    # top p
    top_p: float = None
    # output file
    output_file: str = None

    def _post_init(self) -> None:
        ModelArgs._post_init(self)
        InitializationArgs._post_init(self)
        DatasetArgs._post_init(self)

        # load_path
        if self.load_path is None:
            from src.utils.logging import warn_rank_0

            warn_rank_0("load_path was None, not loading any trained checkpoint")

        # batch_size
        assert self.batch_size is not None, "batch_size cannot be None"

        # max_new_tokens
        assert self.max_new_tokens is not None, "max_new_tokens cannot be None"

        # output_file
        assert self.output_file is not None, "output_file cannot be None"


def get_args(mode: Mode) -> Union[TrainingArgs, InferenceArgs]:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path for the config")
    args = parser.parse_args()

    config: dict = json.load(open(args.config, "r"))

    if mode == Mode.training:
        return TrainingArgs(**config)
    else:
        return InferenceArgs(**config)
