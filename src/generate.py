import json
import os
import sys
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from src.arguments import get_args
from src.constants import DatasetKeys, Mode
from src.dataset import BaseDataset, DatasetSplit
from src.model import Model
from src.utils import ProgressBar, setup_debugging, setup_tf32


def flatten_batch(batch: dict) -> list:
    for any_key in batch.keys():
        break
    batch_size = len(batch[any_key])

    for i in range(batch_size):
        example = {}

        for key in batch.keys():
            if key not in [DatasetKeys.id.value, DatasetKeys.preprocessed_input.value]:
                value = batch[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                example[key] = value

        yield example


def generate(args: Namespace, model: Model, test_dataset: BaseDataset, generate_kwargs: dict) -> None:
    progress_bar = ProgressBar(0, len(test_dataset))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_file = open(args.output_file, "w")

    for batch in test_dataset:
        generated_text = model.generate(batch, generate_kwargs)

        flat_batch = flatten_batch(batch)

        for example, text in zip(flat_batch, generated_text):
            example[DatasetKeys.generated_text.value] = text
            output_file.write(json.dumps(example) + "\n")

        sys.stdout.flush()
        progress_bar.update()


def main() -> None:
    mode = Mode.inference

    setup_tf32()

    args = get_args(mode)

    setup_debugging()

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    model = Model(args, mode)
    if args.load_path is not None:
        model.load_ds_checkpoint(args.load_path)

    test_dataset: BaseDataset = args.data_class(
        args,
        split=DatasetSplit.test,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    # setup model's prepare_input_output_for_generate method from the dataset's implementation
    model.prepare_input_output_for_generate = test_dataset.prepare_input_output_for_generate

    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    generate(args, model, test_dataset, generate_kwargs)


if __name__ == "__main__":
    main()
