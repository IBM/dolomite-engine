from argparse import ArgumentParser, Namespace

import torch
import transformers

from src.checkpointing import ModelCheckpointer
from src.constants import Mode, TrainingInferenceType
from src.model import Model


def get_args() -> Namespace:
    """arguments to use

    Returns:
        Namespace: parsed arguments
    """

    parser = ArgumentParser()

    parser.add_argument("--load_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dtype", type=lambda x: getattr(torch, x), default=torch.float32)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_class", type=lambda x: getattr(transformers, x))
    parser.add_argument(
        "--training_inference_type",
        type=lambda x: getattr(TrainingInferenceType, x),
        choices=[TrainingInferenceType.full_finetuning, TrainingInferenceType.prompt_tuning],
        required=True,
        help="type of tuning, full finetuning or PEFT",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """main program"""

    args = get_args()

    model = Model(args, Mode.inference)
    model.post_init()

    ModelCheckpointer.convert_deepspeed_to_huggingface_checkpoint(model, args.load_path, args.save_path)


if __name__ == "__main__":
    main()
