import glob
import os
from argparse import Namespace
from copy import deepcopy
from typing import List, Set, Type
from uuid import uuid4

import jsonlines
import numpy as np
import torch
from transformers import AutoTokenizer

from src.constants import DUMMY, DatasetKeys, DatasetSplit, Mode, TrainingInferenceType
from src.utils import print_rank_0, register_timer


def check_raw_example(raw_example: dict, mode: Mode) -> None:
    assert (
        DatasetKeys.preprocessed_input.value not in raw_example
    ), "preprocessed_input found in the dataset, please drop this field"
    assert (
        DatasetKeys.preprocessed_output.value not in raw_example
    ), "preprocessed_output found in the dataset, please drop this field"

    if mode == Mode.inference:
        assert (
            DatasetKeys.generated_text.value not in raw_example
        ), "generated_text found in the dataset, please drop this field"


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

        self.do_format_input = self.input_format != "__input__"
        self.do_format_output = self.output_format != "__output__"

        self.max_input_tokens = get_max_input_length(args, is_encoder_decoder)
        self.max_output_tokens = get_max_output_length(args, is_encoder_decoder)

        self.training_inference_type: TrainingInferenceType = args.training_inference_type
        self.num_virtual_tokens: int = args.num_virtual_tokens

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.examples = []

    def construct_input_from_format(self, input: str) -> str:
        if self.do_format_input:
            return self.input_format.replace("__input__", input, 1)
        return input

    def construct_output_from_format(self, output: str) -> str:
        if self.do_format_output:
            return self.output_format.replace("__output__", output, 1)
        return output

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

        return inputs, outputs

    @register_timer("prepare_input_output_for_generate")
    def prepare_input_output_for_generate(self, batch: dict) -> dict:
        inputs = []

        for p in batch[DatasetKeys.preprocessed_input.value]:
            p: List[int] = self.tokenizer(p, add_special_tokens=False)["input_ids"]

            p = p[: self.max_input_tokens]

            if self.is_encoder_decoder:
                p.append(self.tokenizer.eos_token_id)

            inputs.append(p)

        return inputs

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
            DatasetKeys.id.value: generate_random_id(self.__class__),
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
            # don't use uuid here since we want it to be same across epochs and this dataset creates examples on the fly
            DatasetKeys.id.value: str(index),
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

            for raw_example in json_file:
                check_raw_example(raw_example, self.mode)

                result_example = {}

                result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                    raw_example[DatasetKeys.input.value]
                )

                if self.mode == Mode.training:
                    result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                        raw_example[DatasetKeys.output.value]
                    )

                if DatasetKeys.id.value not in raw_example:
                    result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

                result_example.update(raw_example)
                examples.append(result_example)

        return examples


class ConcatenatedDatasets(torch.utils.data.Dataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__()

        self.datasets = self.get_datasets_list(args, split, mode, tokenizer, is_encoder_decoder)
        self.num_examples = sum([len(dataset) for dataset in self.datasets])
        self.start_indices = np.cumsum([0] + [len(dataset) for dataset in self.datasets[:-1]]).tolist()
        self.num_datasets = len(self.datasets)
        self.datasets_key_value_to_add = self.get_dataset_keys()

    def get_dataset_keys(self) -> Set[str]:
        dataset_keys = []
        dataset_key_value_to_add = []
        all_keys = set()

        for dataset in self.datasets:
            example = dataset[0]
            dataset_keys.append(example.keys())
            all_keys.update(example.keys())

        for keys in dataset_keys:
            keys_to_add = all_keys.difference(keys)
            dataset_key_value_to_add.append({k: DUMMY for k in keys_to_add})

        return dataset_key_value_to_add

    @classmethod
    def get_datasets_list(
        cls, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> BaseDataset:
        datasets = []
        for i in range(len(args.data_path)):
            args_copy = deepcopy(args)

            args_copy.data_path = args.data_path[i]
            args_copy.data_class = args.data_class[i]
            args_copy.input_format = args.input_format[i]
            args_copy.output_format = args.output_format[i]
            args_copy.max_input_tokens = args.max_input_tokens[i]
            args_copy.max_output_tokens = args.max_output_tokens[i]

            dataset = args_copy.data_class(args_copy, split, mode, tokenizer, is_encoder_decoder)
            datasets.append(dataset)
        return datasets

    @register_timer("prepare_input_output_for_forward")
    def prepare_input_output_for_forward(self, batch: dict) -> dict:
        batch_subset = {}
        batch_mapping = []

        for p, r, ind in zip(
            batch[DatasetKeys.preprocessed_input.value],
            batch[DatasetKeys.preprocessed_output.value],
            batch[DatasetKeys.data_class_index.value],
        ):
            if ind not in batch_subset:
                batch_subset[ind] = {
                    DatasetKeys.preprocessed_input.value: [],
                    DatasetKeys.preprocessed_output.value: [],
                }

            position = len(batch_subset[ind][DatasetKeys.preprocessed_input.value])
            batch_mapping.append((ind, position))
            batch_subset[ind][DatasetKeys.preprocessed_input.value].append(p)
            batch_subset[ind][DatasetKeys.preprocessed_output.value].append(r)

        for ind in batch_subset:
            dataset_index = int(ind.split(":")[1])
            input_, output_ = self.datasets[dataset_index].prepare_input_output_for_forward(batch_subset[ind])
            batch_subset[ind][DatasetKeys.preprocessed_input.value] = input_
            batch_subset[ind][DatasetKeys.preprocessed_output.value] = output_

        inputs = []
        outputs = []

        for ind, position in batch_mapping:
            input_ = batch_subset[ind][DatasetKeys.preprocessed_input.value][position]
            output_ = batch_subset[ind][DatasetKeys.preprocessed_output.value][position]
            inputs.append(input_)
            outputs.append(output_)

        return inputs, outputs

    @register_timer("prepare_input_output_for_generate")
    def prepare_input_output_for_generate(self, batch: dict) -> dict:
        batch_subset = {}
        batch_mapping = []

        for p, ind in zip(batch[DatasetKeys.preprocessed_input.value], batch[DatasetKeys.data_class_index.value]):
            if ind not in batch_subset:
                batch_subset[ind] = {DatasetKeys.preprocessed_input.value: []}

            position = len(batch_subset[ind][DatasetKeys.preprocessed_input.value])
            batch_mapping.append((ind, position))
            batch_subset[ind][DatasetKeys.preprocessed_input.value].append(p)

        for ind in batch_subset:
            dataset_index = int(ind.split(":")[1])
            input_ = self.datasets[dataset_index].prepare_input_output_for_generate(batch_subset[ind])
            batch_subset[ind][DatasetKeys.preprocessed_input.value] = input_

        inputs = []

        for ind, position in batch_mapping:
            input_ = batch_subset[ind][DatasetKeys.preprocessed_input.value][position]
            inputs.append(input_)

        return inputs

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        dataset_index = self.num_datasets - 1
        for i in range(self.num_datasets):
            if index < self.start_indices[i]:
                dataset_index = i - 1
                break

        index -= self.start_indices[dataset_index]

        example = self.datasets[dataset_index][index]
        example[DatasetKeys.data_class_index.value] = get_data_class_index(
            self.datasets[dataset_index].__class__, dataset_index
        )

        example.update(self.datasets_key_value_to_add[dataset_index])

        return example


def generate_random_id(data_class: BaseDataset) -> str:
    return f"{uuid4()}:{data_class.__name__}"


def get_data_class_index(data_class: BaseDataset, index: int) -> str:
    return f"{data_class.__name__}:{index}"


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
