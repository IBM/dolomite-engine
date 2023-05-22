import json
import os
from typing import List, Union

from transformers import AutoTokenizer

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.data.qa.config import ELI5Config
from src.utils.logging import warn_rank_0


class ELI5Dataset(BaseDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config = ELI5Config(**self.data_config)

        if self.do_format_input:
            raise ValueError(f"input_format for {self.__class__.__name__} should be '__input__'")

        if self.max_input_tokens is None:
            warn_rank_0("ignoring max_document_length in the config since max_input_tokens was not specified")

        self.examples = self.prepare_examples()

    def construct_input_from_format(self, context: str, document: str) -> str:
        context = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        document = self.tokenizer(document, add_special_tokens=False)["input_ids"]

        if self.max_input_tokens is not None:
            context = context[-(self.max_input_tokens - self.data_config.max_document_length) :]
            document = document[: self.data_config.max_document_length - 1]

        input = document + context

        return input

    def construct_output_from_format(self, output: str) -> str:
        output = super().construct_output_from_format(output)
        output = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        output = output[: self.max_output_tokens]
        return output

    def prepare_examples(self) -> List[dict]:
        examples = []
        if self.split.value not in self.data_config.files:
            return examples

        data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])

        with open(data_file, "r") as f:
            json_file = json.load(f)

            for raw_example in json_file:
                check_raw_example(raw_example, self.mode)

                result_example = {}

                # construct input
                context: str = raw_example["context"]
                document: str = raw_example["document"]

                context = context.strip()
                document = document.strip()

                # for decoder only models, we need an explicit marker
                if not self.is_encoder_decoder:
                    context += "\nAgent:"

                result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                    context, document
                )

                # construct output
                if self.mode == Mode.training:
                    output = raw_example["response"]
                    result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(output)

                result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

                result_example["document"] = raw_example["document"]
                result_example["context"] = raw_example["context"]
                result_example["response"] = raw_example["response"]

                examples.append(result_example)

        return examples
