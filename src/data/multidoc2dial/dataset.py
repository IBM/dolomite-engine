import json
import os
from argparse import Namespace
from typing import List

from transformers import AutoTokenizer

from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.data.multidoc2dial.config import DineshChitChatConfig, YatinAnswerabilityConfig


class DineshChitChatDataset(BaseDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config = DineshChitChatConfig(**self.data_config)
        self.examples = self.prepare_examples()

    def tokenize_untokenize(self, context: str, document: str) -> str:
        context = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        document = self.tokenizer(document, add_special_tokens=False)["input_ids"]

        if self.max_input_tokens is not None:
            context = context[-(self.max_input_tokens - 700) :]
            document = document[:700]

        input = document + context
        return self.tokenizer.decode(input)

    def prepare_examples(self) -> List[dict]:
        examples = []
        data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])

        with open(data_file, "r") as f:
            json_file = json.load(f)

            for raw_example in json_file:
                check_raw_example(raw_example, self.mode)

                result_example = {}

                result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                    self.tokenize_untokenize(raw_example["context"], raw_example["document"])
                )

                if self.mode == Mode.training:
                    result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                        raw_example["response"]
                    )

                if DatasetKeys.id.value not in raw_example:
                    result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

                result_example.update(raw_example)
                examples.append(result_example)

        return examples


class YatinAnswerabilityDataset(DineshChitChatDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        BaseDataset.__init__(self, args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config = YatinAnswerabilityConfig(**self.data_config)
        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        examples = super().prepare_examples()

        if not self.data_config.filter:
            return examples

        filtered_examples = []
        for raw_example in examples:
            if raw_example["neg_subtype"].lower() == "original" or raw_example["last_speaker"].lower() == "agent":
                continue
            filtered_examples.append(raw_example)

        return filtered_examples
