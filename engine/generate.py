import json
import os
from typing import List

from .arguments import InferenceArgs, get_args
from .checkpointing import load_checkpoint_for_inference, save_args
from .data import BaseDataset, get_datasets_list
from .enums import DatasetKeys, DatasetSplit, Mode
from .model import Model
from .utils import ProgressBar, setup_tf32


def generate(args: InferenceArgs, model: Model, datasets_list: List[BaseDataset], mode: Mode) -> None:
    """main generation loop

    Args:
        args (InferenceArgs): inference args
        model (Model): non-sharded model
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

    model = Model(args, mode)
    model = model.to(model.input_device)

    datasets_list, _ = get_datasets_list(
        args,
        split=DatasetSplit.test,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    if args.load_args is not None:
        load_checkpoint_for_inference(model, args.load_args.load_path, args.load_args.iteration)

    generate(args, model, datasets_list, mode)


if __name__ == "__main__":
    main()
