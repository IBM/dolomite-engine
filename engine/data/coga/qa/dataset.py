import json
import os
from typing import List, Union

from transformers import AutoTokenizer

from engine.arguments import InferenceArgs, TrainingArgs
from engine.constants import DatasetKeys, DatasetSplit, Mode
from engine.data.coga.filters import filter_dataset
from engine.data.coga.qa.config import ELI5Config
from engine.data.dataset import BaseDataset, check_raw_example, generate_random_id
from engine.utils.logging import print_rank_0, warn_rank_0


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
        self.control_prompt = self.data_config.control_prompt
        print_rank_0("The data name is: ", self.data_name)
        print_rank_0("The control prompt is: ", self.control_prompt)
        if self.control_prompt is None:
            self.control_token_ids = []
        else:
            self.control_token_ids = self.tokenizer(self.control_prompt, add_special_tokens=False)["input_ids"]
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
        if len(self.control_token_ids) != 0:
            input = input + self.control_token_ids

        return input

    def construct_output_from_format(self, output: str) -> str:
        output = super().construct_output_from_format(output)
        output = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        output = output[: self.max_output_tokens]
        return output

    def prepare_examples(self) -> List[dict]:
        if self.split.value not in self.data_config.files:
            return []

        data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])

        with open(data_file, "r") as f:
            raw_examples = json.load(f)
            print_rank_0("Total number of examples in json file: {}".format(len(raw_examples)))
            filtered_examples = filter_dataset(raw_examples)
            print_rank_0("Total number of examples after filtering: {}".format(len(filtered_examples)))
            examples = []
            for raw_example in filtered_examples:
                check_raw_example(raw_example, self.mode)

                result_example = {}

                # construct input
                context: str = raw_example["context"]
                if self.data_config.use_retrieved_passages:
                    passage_similarities: List[float] = raw_example["bertscore_p"]
                    most_similar_passage_index = passage_similarities.index(max(passage_similarities))
                    document: str = raw_example["passage_list"][most_similar_passage_index]
                else:
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

        print_rank_0(f"Loaded {len(examples)} examples from {data_file}")
        return examples
