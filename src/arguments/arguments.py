from argparse import ArgumentParser, Namespace

import torch
import transformers
from peft import PromptTuningInit

import src.data as data_classes
from src.arguments.checkers import verify_args
from src.constants import LearningRateScheduler, Mode, TrainingInferenceType


def get_args(mode: Mode) -> Namespace:
    """arguments to use based on the program mode

    Args:
        mode (Mode): training / inference mode for running the program

    Returns:
        Namespace: arguments based on training / inference mode
    """

    parser = ArgumentParser()

    group = parser.add_argument_group("model")
    group.add_argument("--model_name", type=str, required=True, help="model name on huggingface hub")
    group.add_argument(
        "--model_class",
        type=lambda x: getattr(transformers, x),
        required=True,
        choices=[transformers.AutoModelForCausalLM, transformers.AutoModelForSeq2SeqLM],
        help="model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM",
    )

    group = parser.add_argument_group("checkpointing")
    if mode == Mode.training:
        group.add_argument("--save_path", type=str, required=True, help="path to save checkpoints")
    elif mode == Mode.inference:
        group.add_argument("--load_path", type=str, help="path to load checkpoints")

    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--data_sampling_proportion", type=int, default=[1], nargs="+", help="sampling proportion for the datasets"
    )
    group.add_argument("--data_path", type=str, nargs="+", help="list of datapaths")
    group.add_argument(
        "--data_class", type=lambda x: getattr(data_classes, x), nargs="+", help="list of dataclasses to use"
    )
    group.add_argument("--data_config", type=str, nargs="+", help="list of data configs to use")
    group.add_argument(
        "--input_format", type=str, nargs="+", help="list of format of input in examples in the datasets"
    )
    group.add_argument(
        "--output_format", type=str, nargs="+", help="list of format of output in examples in the datasets"
    )
    group.add_argument("--max_input_tokens", type=int, nargs="+", help="max length for input")
    group.add_argument("--max_output_tokens", type=int, nargs="+", help="max length for output")

    group = parser.add_argument_group("miscellaneous")
    group.add_argument("--seed", type=int, default=42, help="random seed")
    group.add_argument(
        "--dtype",
        type=lambda x: getattr(torch, x),
        choices=[torch.float32, torch.float16, torch.bfloat16],
        help="dtype to use for training / inference",
    )

    # prompt tuning args
    group = parser.add_argument_group("prompt tuning initialization")
    group.add_argument("--prompt_tuning_init", type=lambda x: getattr(PromptTuningInit, x))
    group.add_argument("--prompt_tuning_init_text", type=str)
    group.add_argument("--num_virtual_tokens", type=int)

    group = parser.add_argument_group("training inference")
    group.add_argument(
        "--training_inference_type",
        type=lambda x: getattr(TrainingInferenceType, x),
        choices=[TrainingInferenceType.full_finetuning, TrainingInferenceType.prompt_tuning],
        required=True,
        help="type of tuning, full finetuning or PEFT",
    )

    if mode == Mode.training:
        group = parser.add_argument_group("training")
        group.add_argument("--num_training_steps", type=int, required=True, help="number of training steps")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
        group.add_argument(
            "--eval_and_save_interval", type=int, required=True, help="interval for evaluation and checkpointing"
        )
        group.add_argument("--batch_size_per_gpu", type=int, required=True, help="batch size per GPU for ZeRO-DP")
        group.add_argument("--no_eval", action="store_true", help="avoid evaluating val dataset during training")

        group = parser.add_argument_group("parallelism")
        group.add_argument("--stage", type=int, default=3, help="deepspeed ZeRO stage")
        group.add_argument("--overlap_comm", action="store_true", help="overlap communication with computation")
        group.add_argument(
            "--contiguous_gradients",
            action="store_true",
            default=False,
            help="use contiguous buffers for gradients, requires more memory if enabled",
        )
        group.add_argument("--cpu_offload", action="store_true", help="train with CPU offloading to save GPU memory")

        group = parser.add_argument_group("logging")
        group.add_argument("--logdir", type=str, default="logs", help="logging directory for experiments")

        group = parser.add_argument_group("aim")
        group.add_argument("--disable_aim", action="store_true", help="disable aim logging")
        group.add_argument("--aim_repo", type=str, default="aim_repo", help="aim repo, experiment logs are saved here")
        group.add_argument("--experiment_name", type=str, required=True, help="name of the experiment")

        group = parser.add_argument_group("optimizer and scheduler")
        group.add_argument("--learning_rate", type=float, default=1e-5)
        group.add_argument("--weight_decay", type=float, default=0.1)
        group.add_argument("--beta1", type=float, default=0.9)
        group.add_argument("--beta2", type=float, default=0.95)
        group.add_argument("--eps", type=float, default=1e-8)
        group.add_argument("--warmup_steps", type=int, default=200)
        group.add_argument(
            "--lr_schedule",
            type=lambda x: getattr(LearningRateScheduler, x),
            choices=[LearningRateScheduler.linear, LearningRateScheduler.cosine],
            help="learning rate schedule",
        )

        group = parser.add_argument_group("debug")
        group.add_argument("--steps_per_print", type=int, default=10, help="steps per print")
    else:
        group = parser.add_argument_group("inference")
        group.add_argument("--batch_size", type=int, required=True, help="batch size")
        group.add_argument("--do_sample", action="store_true", help="sample or greedy")
        group.add_argument("--max_new_tokens", type=int, default=20, help="max new tokens")
        group.add_argument("--temperature", type=float, help="temperature")
        group.add_argument("--top_k", type=int, help="top k")
        group.add_argument("--top_p", type=float, help="top p")

        group = parser.add_argument_group("output")
        group.add_argument("--output_file", type=str, help="output file")

    args = parser.parse_args()

    verify_args(args)

    return args
