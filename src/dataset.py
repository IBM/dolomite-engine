import glob
import os
from argparse import Namespace
from typing import List, Type

import jsonlines
import torch
from transformers import AutoTokenizer

from src.constants import DatasetKeys, DatasetSplit, Mode, TrainingInferenceType
from src.utils import print_rank_0, register_timer


def pad(arrays: list, padding: int, max_length: int = None, side: str = "left"):
    if max_length is None:
        max_length = max(list(map(len, arrays)))

    if side == "left":
        inputs = [[padding] * (max_length - len(array)) + array for array in arrays]
        masks = [[0] * (max_length - len(array)) + [1] * len(array) for array in arrays]
    else:
        inputs = [array + [padding] * (max_length - len(array)) for array in arrays]
        masks = [[1] * len(array) + [0] * (max_length - len(array)) for array in arrays]

    return inputs, masks


def get_max_input_length(args: Namespace, is_encoder_decoder: bool) -> int:
    if args.max_input_tokens is None:
        return None

    if is_encoder_decoder:
        if args.training_inference_type == TrainingInferenceType.full_finetuning:
            return args.max_input_tokens - 1
        elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
            return args.max_input_tokens - args.num_virtual_tokens - 1
    else:
        if args.training_inference_type == TrainingInferenceType.full_finetuning:
            return args.max_input_tokens
        elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
            return args.max_input_tokens - args.num_virtual_tokens


def get_max_output_length(args: Namespace, is_encoder_decoder: bool) -> int:
    if args.max_output_tokens is None:
        return None

    if is_encoder_decoder:
        if args.training_inference_type == TrainingInferenceType.full_finetuning:
            return args.max_output_tokens - 1
        elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
            return args.max_output_tokens - args.num_virtual_tokens - 1
    else:
        return args.max_output_tokens - 1


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__()

        self.data_path: str = args.data_path
        self.split = split
        self.mode = mode

        self.input_format: str = args.input_format
        self.output_format: str = args.output_format

        self.max_input_tokens = get_max_input_length(args, is_encoder_decoder)
        self.max_output_tokens = get_max_output_length(args, is_encoder_decoder)

        self.training_inference_type: TrainingInferenceType = args.training_inference_type
        self.num_virtual_tokens: int = args.num_virtual_tokens

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.examples = []

    def construct_input_from_format(self, input: str) -> str:
        return self.input_format.replace("__input__", input, 1)

    def construct_output_from_format(self, output: str) -> str:
        return self.output_format.replace("__output__", output, 1)

    @register_timer("prepare_input_output_for_forward")
    def prepare_input_output_for_forward(self, batch: dict) -> dict:
        inputs = []
        outputs = []

        for p, r in zip(batch[DatasetKeys.preprocessed_input.value], batch[DatasetKeys.preprocessed_output.value]):
            p: List[int] = self.tokenizer(p, add_special_tokens=False)["input_ids"]
            r: List[int] = self.tokenizer(r, add_special_tokens=False)["input_ids"]

            p = p[: self.max_input_tokens]
            r = r[: self.max_output_tokens]

            r.append(self.tokenizer.eos_token_id)
            outputs.append(r)

            if self.is_encoder_decoder:
                p.append(self.tokenizer.eos_token_id)
                inputs.append(p)
            else:
                pr = p + r
                inputs.append(pr)

        max_length = None
        if not self.is_encoder_decoder:
            max_length = max(list(map(len, inputs)))

        input_ids, attention_mask = pad(
            inputs, padding=self.tokenizer.pad_token_id, max_length=max_length, side=self.tokenizer.padding_side
        )
        labels, _ = pad(outputs, padding=-100, max_length=max_length, side=self.tokenizer.padding_side)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    @register_timer("prepare_input_output_for_generate")
    def prepare_input_output_for_generate(self, batch: dict) -> dict:
        inputs = []

        for p in batch[DatasetKeys.preprocessed_input.value]:
            p: List[int] = self.tokenizer(p, add_special_tokens=False)["input_ids"]

            p = p[: self.max_input_tokens]

            if self.is_encoder_decoder:
                p.append(self.tokenizer.eos_token_id)

            inputs.append(p)

        input_ids, attention_mask = pad(inputs, padding=self.tokenizer.pad_token_id, side=self.tokenizer.padding_side)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


class DebugDataset(BaseDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.max_input_tokens: int = args.max_input_tokens
        self.max_output_tokens: int = args.max_output_tokens

    def __getitem__(self, index: int) -> dict:
        example = {
            DatasetKeys.id.value: str(index),
            DatasetKeys.input.value: " Hello" * self.max_input_tokens,
            DatasetKeys.output.value: " Hello" * self.max_output_tokens,
        }
        example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
            example[DatasetKeys.input.value]
        )
        example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
            example[DatasetKeys.output.value]
        )
        return example

    def __len__(self) -> int:
        return 100


class SST2Dataset(BaseDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        split = split.value
        if split == "val":
            split = "validation"

        from datasets import load_dataset

        self.examples = load_dataset("sst2")[split]
        print_rank_0(f"{len(self.examples)} examples in {self.split.value} split")

    def __getitem__(self, index: int) -> dict:
        raw_example = self.examples[index]

        result_example = {
            DatasetKeys.id.value: index,
            DatasetKeys.input.value: raw_example["sentence"].strip(),
            DatasetKeys.output.value: raw_example["label"],
        }

        result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
            result_example[DatasetKeys.input.value]
        )

        if self.mode == Mode.training:
            result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                "positive" if result_example[DatasetKeys.output.value] == 1 else "negative"
            )

        return result_example


class JSONLinesDataset(BaseDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.examples = self.prepare_examples()
        print_rank_0(f"{len(self.examples)} examples in {self.split.value} split")

    def prepare_examples(self) -> List[dict]:
        examples = []
        data_files = glob.glob(os.path.join(self.data_path, self.split.value, "*.jsonl"))

        for filename in data_files:
            json_file = jsonlines.open(filename, "r")
            id = 0

            for raw_example in json_file:
                result_example = {
                    DatasetKeys.id.value: f"{filename}-{id}",
                    DatasetKeys.input.value: raw_example[DatasetKeys.input.value],
                }

                result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                    result_example[DatasetKeys.input.value]
                )

                if DatasetKeys.output.value in raw_example:
                    result_example[DatasetKeys.output.value] = raw_example[DatasetKeys.output.value]

                if self.mode == Mode.training:
                    result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                        result_example[DatasetKeys.output.value]
                    )

                result_example.update(raw_example)
                examples.append(result_example)
                id += 1

        return examples


def get_dataset_class(data_class: str) -> Type[BaseDataset]:
    from sys import modules

    return getattr(modules[__name__], data_class)
