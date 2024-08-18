import json
import os

import torch

from .arguments import InferenceArgs, get_args
from .checkpointing import load_checkpoint_for_inference, save_args
from .data import BaseDataset, collate_fn, get_datasets_list
from .enums import DatasetKeys, DatasetSplit, Mode, TuningMethod
from .model_wrapper import ModelWrapper, ModelWrapperForFinetuning
from .utils import ProcessGroupManager, ProgressBar, setup_tf32


def generate(args: InferenceArgs, model: ModelWrapper, datasets_list: list[BaseDataset], mode: Mode) -> None:
    """main generation loop

    Args:
        args (InferenceArgs): inference args
        model (ModelWrapper): non-sharded model
        datasets_list (list[BaseDataset]): list of datasets
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
                batch = collate_fn(
                    batch,
                    mode=mode,
                    loss_mask=None,
                    eos_token_id=model.eos_token_id,
                    is_encoder_decoder=model.is_encoder_decoder,
                    use_padding_free_transformer=False,
                )

                generated_text, num_generated_tokens = model.generate(batch, generate_kwargs)

                for generated_text_, num_generated_tokens_ in zip(generated_text, num_generated_tokens):
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
        assert not args.model_args.efficient_initialization
        assert not args.model_args.use_padding_free_transformer

        with (
            torch.device(torch.cuda.current_device()),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ):
            model = ModelWrapperForFinetuning(
                mode=mode,
                model_name=args.model_args.model_name,
                pretrained_config=args.model_args.pretrained_config,
                model_class=args.model_args.model_class,
                dtype=args.mixed_precision_args.dtype,
                efficient_initialization=False,
                attention_implementation=args.model_args.attention_implementation,
                use_padding_free_transformer=False,
                tensor_parallel_word_embeddings=False,
                sequence_parallel=False,
                distributed_backend=None,
                random_seed=args.random_args.seed,
                neft_alpha=None,
                trust_remote_code=args.model_args.trust_remote_code,
                tokenizer_name=args.tokenizer_args.tokenizer_name,
                additional_special_tokens=args.tokenizer_args.additional_special_tokens,
            )

        datasets_list, _ = get_datasets_list(
            dataset_args_list=args.datasets,
            tuning_method=TuningMethod.full_finetuning,
            split=DatasetSplit.test,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )
    else:
        model, args_from_checkpoint, _ = load_checkpoint_for_inference(args, mode)

        tuning_method = args_from_checkpoint.tuning_args.tuning_method

        datasets_list, _ = get_datasets_list(
            dataset_args_list=args.datasets,
            tuning_method=tuning_method,
            split=DatasetSplit.test,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
            num_virtual_tokens=args_from_checkpoint.tuning_args.get_num_virtual_tokens(),
        )

    model = model.to(torch.cuda.current_device())

    generate(args, model, datasets_list, mode)

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
