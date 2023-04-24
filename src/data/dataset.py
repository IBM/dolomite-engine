import glob
import os
from copy import deepcopy
from typing import List, Set, Tuple, Type, Union
from uuid import uuid4

import jsonlines
import numpy as np
import torch
from transformers import AutoTokenizer

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import DUMMY, DatasetConfigKeys, DatasetKeys, DatasetSplit, Mode, TrainingInferenceType
from src.utils import print_rank_0, register_timer


class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset class to be implemented by all the datasets"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.training_inference_type: TrainingInferenceType = args.training_inference_type
        # used for prompt tuning
        self.num_virtual_tokens: int = args.num_virtual_tokens

        data_config: dict = args.data_config

        self.data_name: str = data_config.get(DatasetConfigKeys.data_name.value)
        self.data_path: str = data_config.get(DatasetConfigKeys.data_path.value)

        self.input_format: str = data_config.get(DatasetConfigKeys.input_format.value, "__input__")
        self.output_format: str = data_config.get(DatasetConfigKeys.output_format.value, "__output__")

        # if format is __input__ or __output__ formatting is a no-op
        self.do_format_input = self.input_format != "__input__"
        self.do_format_output = self.output_format != "__output__"

        # length to use for trimming
        self.max_input_tokens = get_max_input_length(
            data_config.get(DatasetConfigKeys.max_input_tokens.value),
            self.training_inference_type,
            self.num_virtual_tokens,
            self.is_encoder_decoder,
        )
        self.max_output_tokens = get_max_output_length(
            data_config.get(DatasetConfigKeys.max_output_tokens.value),
            self.training_inference_type,
            self.num_virtual_tokens,
            self.is_encoder_decoder,
        )

        self.data_config: dict = drop_common_args(deepcopy(data_config))

        self.examples = []

    def construct_input_from_format(self, input: str) -> str:
        """construct input with the specified input_format

        Args:
            input (str): input text

        Returns:
            str: formatted text
        """

        if self.do_format_input:
            return self.input_format.replace("__input__", input, 1)
        return input

    def construct_output_from_format(self, output: str) -> str:
        """construct output with the specified output_format

        Args:
            output (str): output text

        Returns:
            str: formatted text
        """

        if self.do_format_output:
            return self.output_format.replace("__output__", output, 1)
        return output

    def collate_fn(self, batch: List[dict]) -> Union[List[List[int]], Tuple[List[List[int]]]]:
        """prepares the inputs and outputs for forward function depending on whther the model is decoder only or encoder-decoder

        Args:
            batch (List[dict]): batch of examples

        Returns:
            Union[List[List[int]], Tuple[List[List[int]]]]: batch of token ids
        """

        inputs = []
        outputs = []

        for example in batch:
            p: Union[str, List[int]] = example[DatasetKeys.preprocessed_input.value]

            # check if not pre-tokenized
            if isinstance(p, str):
                p = self.tokenizer(p, add_special_tokens=False)["input_ids"]

            if self.max_input_tokens is not None:
                if self.is_encoder_decoder:
                    # eos needs to be added later
                    p = p[: self.max_input_tokens - 1]
                else:
                    p = p[: self.max_input_tokens]

            if self.mode == Mode.training:
                r: Union[str, List[int]] = example[DatasetKeys.preprocessed_output.value]

                # check if not pre-tokenized
                if isinstance(r, str):
                    r = self.tokenizer(r, add_special_tokens=False)["input_ids"]

                if self.max_output_tokens is not None:
                    r = r[: self.max_output_tokens - 1]

                r.append(self.tokenizer.eos_token_id)
                outputs.append(r)

                if self.is_encoder_decoder:
                    p.append(self.tokenizer.eos_token_id)
                    inputs.append(p)
                else:
                    inputs.append(p + r)
            else:
                if self.is_encoder_decoder:
                    p.append(self.tokenizer.eos_token_id)

                inputs.append(p)

        if self.mode == Mode.training:
            return inputs, outputs
        else:
            return inputs

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        if self.data_name is not None:
            example[DatasetConfigKeys.data_name.value] = self.data_name
        return example

    def __len__(self) -> int:
        return len(self.examples)


class DebugDataset(BaseDataset):
    """A dummy dataset for profiling and timing the code"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
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

        if self.mode == Mode.training:
            example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                example[DatasetKeys.output.value]
            )

        return example

    def __len__(self) -> int:
        return 100


class SST2Dataset(BaseDataset):
    """SST2 dataset for sentiment classification"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        split = split.value
        if split == "val":
            split = "validation"

        from datasets import load_dataset

        self.examples = load_dataset("sst2")[split]
        print_rank_0(f"{len(self.examples)} examples in {self.split.value} split")

    def __getitem__(self, index: int) -> dict:
        raw_example = super().__getitem__(index)

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
    """A dataset for loading JSON lines files"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
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
    """Concatenated list of datasets for training or inference"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.datasets, self.data_sampling_proportion = self.get_datasets_list(args)

        num_examples_in_each_dataset = self.get_num_examples_in_each_dataset()
        self.num_examples = sum(num_examples_in_each_dataset)
        self.start_indices = np.cumsum([0] + num_examples_in_each_dataset[:-1]).tolist()

        self.datasets_key_value_to_add = self.get_dataset_keys()

        self.print_dataset_stats()

    def get_dataset_keys(self) -> Set[str]:
        """gets the combined set of keys in each dataset

        Returns:
            Set[str]: set of keys in all the datasets
        """

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

    def get_datasets_list(self, args: Union[TrainingArgs, InferenceArgs]) -> Tuple[List[BaseDataset], List[int]]:
        """prepare all the datasets

        Args:
            args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
            split (DatasetSplit): dataset split to use
            mode (Mode): training / inference mode
            tokenizer (AutoTokenizer): tokenizer to use
            is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

        Returns:
            Tuple[List[BaseDataset], List[int]]: list of all datasets, data sampling proportion
        """

        datasets = []
        data_sampling_proportion = []

        for data_config in args.datasets:
            args_copy = deepcopy(args)
            args_copy.data_config = deepcopy(data_config)
            del args_copy.datasets

            dataset = args_copy.data_config[DatasetConfigKeys.data_class.value](
                args_copy, self.split, self.mode, self.tokenizer, self.is_encoder_decoder
            )

            if len(dataset) > 0:
                datasets.append(dataset)
                data_sampling_proportion.append(data_config[DatasetConfigKeys.data_sampling_proportion.value])

        return datasets, data_sampling_proportion

    @register_timer("collate_fn")
    def collate_fn(self, batch: List[dict]) -> Union[List[List[int]], Tuple[List[List[int]]]]:
        """applies collate_fn to each example depending on which dataset they come from

        Args:
            batch (dict): batch of examples

        Returns:
            Union[List[List[int]], Tuple[List[List[int]]]]: batch of token ids
        """

        batch_subset = {}
        batch_mapping = []

        for example in batch:
            ind: str = example[DatasetKeys.data_class_index.value]

            if ind not in batch_subset:
                batch_subset[ind] = []

            position = len(batch_subset[ind])
            batch_mapping.append((ind, position))
            batch_subset[ind].append(example)

        for ind in batch_subset:
            dataset_index = int(ind.split(":")[1])
            batch_subset[ind] = self.datasets[dataset_index].collate_fn(batch_subset[ind])

        inputs = []
        outputs = []

        for ind, position in batch_mapping:
            if self.mode == Mode.training:
                input = batch_subset[ind][0][position]
                output = batch_subset[ind][1][position]

                outputs.append(output)
            else:
                input = batch_subset[ind][position]

            inputs.append(input)

        if self.mode == Mode.training:
            return inputs, outputs
        else:
            return inputs

    def get_num_datasets(self) -> int:
        """returns the number of datasets in the mixture

        Returns:
            int: number of datasets in the mixture
        """

        return len(self.datasets)

    def get_num_examples_in_each_dataset(self) -> List[int]:
        """returns the number of examples in each dataset component

        Returns:
            List[int]: the number of examples in each dataset component
        """

        return [len(dataset) for dataset in self.datasets]

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        num_datasets = self.get_num_datasets()

        # get the dataset the example belongs to
        dataset_index = num_datasets - 1
        for i in range(num_datasets):
            if index < self.start_indices[i]:
                dataset_index = i - 1
                break

        # get the position of the example in the specific dataset
        index -= self.start_indices[dataset_index]

        # get the example
        example = self.datasets[dataset_index][index]
        example[DatasetKeys.data_class_index.value] = get_data_class_index(
            self.datasets[dataset_index].__class__, dataset_index
        )

        # update the example with dummy values if the some specific keys are absent in a dataset
        example.update(self.datasets_key_value_to_add[dataset_index])

        return example

    def print_dataset_stats(self) -> None:
        """prints the statistics of all the datasets"""

        print_rank_0(f"{'-' * 25} {self.split.value} {'-' * 25}")

        print_rank_0(f"number of datasets = {self.get_num_datasets()}")
        print_rank_0(f"total examples in the entire dataset mixture = {len(self)}\n")

        for dataset in self.datasets:
            print_rank_0(f"examples in {dataset.__class__.__name__} = {len(dataset)}")

        print_rank_0("-" * 57)


def generate_random_id(data_class: Type[BaseDataset]) -> str:
    """generates a random unique ID for every example

    Args:
        data_class (Type[BaseDataset]): dataset class to which the example belongs

    Returns:
        str: randomly generated ID
    """

    return f"{uuid4()}:{data_class.__name__}"


def get_data_class_index(data_class: Type[BaseDataset], index: int) -> str:
    """get dataset label in the list (concatnated name and index in list)

    Args:
        data_class (Type[BaseDataset]): specific data class
        index (int): position in the list

    Returns:
        str: unique dataset label in the list
    """

    return f"{data_class.__name__}:{index}"


def get_max_input_length(
    max_input_tokens_specified: int,
    training_inference_type: TrainingInferenceType,
    num_virtual_tokens: int,
    is_encoder_decoder: bool,
) -> int:
    """max input length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_input_tokens_specified (int): maximum number of specified input tokens
        training_inference_type (TrainingInferenceType): full finetuning / prompt tuning
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max input length
    """

    if max_input_tokens_specified is None:
        return None

    if is_encoder_decoder:
        if training_inference_type == TrainingInferenceType.full_finetuning:
            return max_input_tokens_specified - 1
        elif training_inference_type == TrainingInferenceType.prompt_tuning:
            return max_input_tokens_specified - num_virtual_tokens - 1
    else:
        if training_inference_type == TrainingInferenceType.full_finetuning:
            return max_input_tokens_specified
        elif training_inference_type == TrainingInferenceType.prompt_tuning:
            return max_input_tokens_specified - num_virtual_tokens


def get_max_output_length(
    max_output_tokens_specified: int,
    training_inference_type: TrainingInferenceType,
    num_virtual_tokens: int,
    is_encoder_decoder: bool,
) -> int:
    """max output length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_output_tokens_specified (int): maximum number of specified output tokens
        training_inference_type (TrainingInferenceType): full finetuning / prompt tuning
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max output length
    """

    if max_output_tokens_specified is None:
        return None

    if is_encoder_decoder:
        if training_inference_type == TrainingInferenceType.full_finetuning:
            return max_output_tokens_specified - 1
        elif training_inference_type == TrainingInferenceType.prompt_tuning:
            return max_output_tokens_specified - num_virtual_tokens - 1
    else:
        return max_output_tokens_specified - 1


def check_raw_example(raw_example: dict, mode: Mode) -> None:
    """checks whether the dataset has conflicting fields

    Args:
        raw_example (dict): example to check
        mode (Mode): training / inference mode for running the program
    """

    assert (
        DatasetKeys.preprocessed_input.value not in raw_example
    ), "preprocessed_input found in the dataset, please drop this field"
    assert (
        DatasetKeys.preprocessed_output.value not in raw_example
    ), "preprocessed_output found in the dataset, please drop this field"
    assert (
        DatasetKeys.data_class_index.value not in raw_example
    ), "data_class_index found in the dataset, please drop this field"

    if mode == Mode.inference:
        assert (
            DatasetKeys.generated_text.value not in raw_example
        ), "generated_text found in the dataset, please drop this field"


def drop_common_args(data_config: dict) -> dict:
    for key in DatasetConfigKeys:
        if key.value in data_config:
            del data_config[key]

    return data_config
