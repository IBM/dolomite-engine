import json
import os
import sys
from argparse import Namespace

from src.arguments import get_args
from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data import ConcatenatedDatasets
from src.model import Model
from src.utils import ProgressBar, setup_debugging, setup_tf32


def generate(args: Namespace, model: Model, test_dataset: ConcatenatedDatasets, generate_kwargs: dict) -> None:
    """main generation loop

    Args:
        args (Namespace): inference args
        model (Model): non-sharded model
        test_dataset (ConcatenatedDatasets): test dataset
        generate_kwargs (dict): generation arguments
    """

    progress_bar = ProgressBar(0, len(test_dataset))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_file = open(args.output_file, "w")
    raw_batch = []

    for index, example in enumerate(test_dataset):
        raw_batch.append(example)

        if len(raw_batch) == args.batch_size or index == len(test_dataset) - 1:
            batch = test_dataset.collate_fn(raw_batch)
            generated_text, num_generated_tokens = model.generate(batch, generate_kwargs)

            for example, generated_text_, num_generated_tokens_ in zip(
                raw_batch, generated_text, num_generated_tokens
            ):
                example[DatasetKeys.generated_text.value] = generated_text_
                example[DatasetKeys.num_generated_tokens.value] = num_generated_tokens_

                del example[DatasetKeys.preprocessed_input.value]

                output_file.write(json.dumps(example) + "\n")
                sys.stdout.flush()

            raw_batch = []

        progress_bar.update()


def main() -> None:
    """main program"""

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

    test_dataset = ConcatenatedDatasets(
        args,
        split=DatasetSplit.test,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    model.post_init()
    if args.load_path is not None:
        model.load_ds_checkpoint(args.load_path)

    generate(args, model, test_dataset, generate_kwargs)


if __name__ == "__main__":
    main()
