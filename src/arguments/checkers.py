from argparse import Namespace
from typing import List

from peft import PromptTuningInit

import src.data as data_classes
from src.constants import DatasetConfigKeys, TrainingInferenceType


def check_training_inference_type(args: Namespace) -> None:
    """checks whether the arguments specified are valid for finetuning / prompt tuning

    Args:
        args (Namespace): arguments based on training / inference mode
    """

    if args.training_inference_type == TrainingInferenceType.full_finetuning:
        assert (
            args.prompt_tuning_init is None
        ), f"prompt_tuning_init '{args.prompt_tuning_init}' should not be specified with full_finetuning"
        assert (
            args.prompt_tuning_init_text is None
        ), f"prompt_tuning_init_text '{args.prompt_tuning_init_text}' should not be specified with full_finetuning"
        assert (
            args.num_virtual_tokens is None
        ), f"num_virtual_tokens '{args.num_virtual_tokens}' should not be specified with full_finetuning"
    elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
        if args.prompt_tuning_init == PromptTuningInit.RANDOM:
            assert (
                args.prompt_tuning_init_text is None
            ), f"prompt_tuning_init_text '{args.prompt_tuning_init_text}' was specified with RANDOM init method"
        elif args.prompt_tuning_init == PromptTuningInit.TEXT:
            assert (
                args.prompt_tuning_init_text is not None
            ), f"prompt_tuning_init_text needs to be specified with TEXT init method"


def check_dataset_configs_json(dataset_configs_json: List[dict]) -> None:
    """checks whether the arguments specified in the config are valid

    Args:
        dataset_configs_json (List[dict]): loaded json config
    """

    for i, data_config in enumerate(dataset_configs_json):
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
