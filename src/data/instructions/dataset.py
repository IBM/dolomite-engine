import json
import os
from typing import List, Union

import jsonlines
from transformers import AutoTokenizer

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.data.instructions.config import AlpacaConfig, DollyConfig, VicunaConfig
from src.data.utils import train_val_test_split


class BaseInstructionDataset(BaseDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        if self.do_format_input:
            raise ValueError(f"input_format for {self.__class__.__name__} should be '__input__'")

    def construct_input_from_format(self, instruction: str, input: str) -> List[int]:
        input_text = instruction + "\n\n"
        if not (input is None or input == ""):
            input_text += f"input: {input}\n"
        input_text += "output:"
        return input_text


class AlpacaDataset(BaseInstructionDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config = AlpacaConfig(**self.data_config)
        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        data = json.load(open(os.path.join(self.data_path, "alpaca_data.json"), "r"))
        data = train_val_test_split(
            data, self.split, self.data_config.seed, self.data_config.val_samples, self.data_config.test_samples
        )

        examples = []
        for raw_example in data:
            raw_example: dict
            check_raw_example(raw_example, mode=self.mode)

            result_example = {}
            result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                raw_example["instruction"], raw_example.get("input", "")
            )

            if self.mode == Mode.training:
                result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                    raw_example[DatasetKeys.output.value].strip()
                )

            if DatasetKeys.id.value not in raw_example:
                result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

            result_example.update(raw_example)
            examples.append(result_example)

        return examples


class DollyDataset(BaseInstructionDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config = DollyConfig(**self.data_config)
        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        data = [i for i in jsonlines.open(os.path.join(self.data_path, "databricks-dolly-15k.jsonl"), "r")]
        data = train_val_test_split(
            data, self.split, self.data_config.seed, self.data_config.val_samples, self.data_config.test_samples
        )

        examples = []
        for raw_example in data:
            raw_example: dict
            check_raw_example(raw_example, mode=self.mode)

            result_example = {}
            result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                raw_example["instruction"], raw_example.get("context", "")
            )

            if self.mode == Mode.training:
                result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                    raw_example["response"].strip()
                )

            if DatasetKeys.id.value not in raw_example:
                result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

            result_example.update(raw_example)
            examples.append(result_example)

        return examples


class VicunaDataset(BaseDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)
        self.data_config = VicunaConfig(**self.data_config)

        if self.do_format_input:
            raise ValueError(f"input_format for {self.__class__.__name__} should be '__input__'")

        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        if self.data_config.filter_sorry:
            data_path = os.path.join(self.data_path, "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
        else:
            data_path = os.path.join(self.data_path, "ShareGPT_V3_unfiltered_cleaned_split.json")

        data = json.load(open(data_path))
        data = train_val_test_split(
            data, self.split, self.data_config.seed, self.data_config.val_samples, self.data_config.test_samples
        )

        examples = []
        for raw_conversation in data:
            joined_conversation = self.join_consecutive_utterances(raw_conversation["conversations"])

            context = ""
            for utterance in joined_conversation:
                if utterance["from"] == "gpt":
                    result_example = {
                        DatasetKeys.preprocessed_input.value: f"{context}\nAgent:".strip(),
                        DatasetKeys.id.value: generate_random_id(self.__class__),
                    }

                    if self.mode == Mode.training:
                        result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                            utterance["value"]
                        )

                    examples.append(result_example)
                else:
                    context += f"\nUser: {utterance['value']}"

        return examples

    def join_consecutive_utterances(self, conversation: List[dict]) -> List[dict]:
        if len(conversation) < 2:
            return conversation

        i = 0
        while i < len(conversation) - 1:
            if conversation[i]["from"] == conversation[i + 1]["from"]:
                conversation[i]["value"] += conversation[i + 1]["value"]
                del conversation[i + 1]
            else:
                i += 1

        return conversation
