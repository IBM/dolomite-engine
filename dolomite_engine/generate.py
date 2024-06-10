import json
import os
from typing import List

import torch

from .arguments import InferenceArgs, get_args
from .checkpointing import load_checkpoint_for_inference, save_args
from .data import BaseDataset, get_datasets_list
from .enums import DatasetKeys, DatasetSplit, Mode
from .model_wrapper import ModelWrapper, get_model
from .utils import ProgressBar, setup_tf32


def generate(args: InferenceArgs, model: ModelWrapper, datasets_list: List[BaseDataset], mode: Mode) -> None:
    """main generation loop

    Args:
        args (InferenceArgs): inference args
        model (ModelWrapper): non-sharded model
        datasets_list (List[BaseDataset]): list of datasets
        mode (Mode): training / inference mode
    """

    batch_size = args.generation_parameters.batch_size

    progress_bar = ProgressBar(0, sum([len(dataset) for dataset in datasets_list]))

    os.makedirs(args.output_dir, exist_ok=True)
    save_args(args, args.output_dir, mode)

    generate_kwargs = args.generation_parameters.to_dict()
    generate_kwargs.pop("batch_size")

    for dataset in datasets_list:
        output_file = open(os.path.join(args.output_dir, f"output-{dataset.data_name}.jsonl"), "w")
        batch = []

        for index, example in enumerate(dataset):
            batch.append(example)

            if len(batch) == batch_size or index == len(dataset) - 1:
                generated_text, num_generated_tokens = model.generate(batch, generate_kwargs)

                for example, generated_text_, num_generated_tokens_ in zip(
                    batch, generated_text, num_generated_tokens
                ):
                    output_file.write(
                        json.dumps(
                            {
                                DatasetKeys.generated_text.value: generated_text_,
                                DatasetKeys.num_generated_tokens.value: num_generated_tokens_,
                            }
                        )
                        + "\n"
                    )

                batch = []

            progress_bar.update()


def main() -> None:
    """main program"""

    mode = Mode.inference

    setup_tf32()

    args: InferenceArgs = get_args(mode)

    # hardcoded single GPU assumed for inference
    torch.cuda.set_device(0)

    if args.load_args is None:
        model = get_model(args, mode)
        model = model.to(torch.cuda.current_device())

        datasets_list, _ = get_datasets_list(
            args,
            split=DatasetSplit.test,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )
    else:
        model, args_from_checkpoint = load_checkpoint_for_inference(args, mode)

        # override with datasets passed in current config
        args_from_checkpoint.datasets = args.datasets

        datasets_list, _ = get_datasets_list(
            args_from_checkpoint,
            split=DatasetSplit.test,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )

    generate(args, model, datasets_list, mode)


if __name__ == "__main__":
    main()
