from argparse import Namespace

from peft import PromptTuningInit

from src.constants import TrainingInferenceType
from src.data import DebugDataset
from src.utils import is_debugging_enabled, warn_rank_0


def check_training_inference_type(args: Namespace) -> None:
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


def set_input_defaults(args: Namespace) -> None:
    if args.max_input_tokens is None:
        args.max_input_tokens = [None] * len(args.data_sampling_proportion)
        warn_rank_0("max_input_tokens was not specified, setting to inf")

    if args.input_format is None:
        args.input_format = ["__input__"] * len(args.data_sampling_proportion)
        warn_rank_0(
            f"input_format was not specified and setting to the default value '__input__' for all the datasets"
        )


def set_output_defaults(args: Namespace) -> None:
    if args.max_output_tokens is None:
        args.max_output_tokens = [None] * len(args.data_sampling_proportion)
        warn_rank_0("max_output_tokens was not specified, setting to inf")

    if args.output_format is None:
        args.output_format = ["__output__"] * len(args.data_sampling_proportion)
        warn_rank_0(
            f"output_format was not specified and setting to the default value '__output__' for all the datasets"
        )


def check_dataset_args(args: Namespace) -> None:
    data_args_to_check = [
        "data_sampling_proportion",
        "data_path",
        "data_class",
        "input_format",
        "output_format",
        "max_input_tokens",
        "max_output_tokens",
    ]
    for i in range(1, len(data_args_to_check)):
        l1 = len(getattr(args, data_args_to_check[0]))
        l2 = len(getattr(args, data_args_to_check[i]))
        assert (
            l1 == l2
        ), f"{data_args_to_check[0]} and {data_args_to_check[i]} are lists of different lengths {l1} and {l2} respectively"


def check_debugging_args(args: Namespace) -> None:
    if is_debugging_enabled():
        if args.data_class != DebugDataset:
            warn_rank_0(f"found data_class '{args.data_class}'. For debugging, DebugDataset is recommended")

    if args.data_class == DebugDataset:
        if args.input_format != ["__input__"] * len(args.data_sampling_proportion):
            warn_rank_0(
                f"ignoring the specified input_format '{args.input_format}' and setting to the default value '__input__'"
            )
            args.input_format = ["__input__"] * len(args.data_sampling_proportion)

        if args.output_format != ["__output__"] * len(args.data_sampling_proportion):
            warn_rank_0(
                f"ignoring the specified output_format '{args.output_format}' and setting to the default value '__output__'"
            )
            args.output_format = ["__output__"] * len(args.data_sampling_proportion)


def verify_args(args: Namespace) -> None:
    check_training_inference_type(args)

    set_input_defaults(args)
    set_output_defaults(args)

    check_dataset_args(args)

    check_debugging_args(args)
