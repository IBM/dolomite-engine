import json
import os
from argparse import Namespace

from transformers import AutoTokenizer

from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.utils.logging import warn_rank_0


class AlpacaDataset(BaseDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        if split != DatasetSplit.train:
            raise NotImplementedError("only train split is support in this dataset")

        if self.do_format_input:
            raise NotImplementedError(f"{self.__class__.__name__} doesn't support input_format argument")
        else:
            warn_rank_0(
                f"{self.__class__.__name__} has its own input formatting and ignores the specified input_format"
            )

        self.prompt_format = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        self.examples = self.prepare_examples(args.data_path)

    def prepare_examples(self, data_path: str) -> list:
        data = json.load(open(os.path.join(data_path, "alpaca_data.json"), "r"))

        examples = []
        for raw_example in data:
            check_raw_example(raw_example, mode=self.mode)

            result_example = {}

            if raw_example.get("input", "") != "":
                result_example[DatasetKeys.preprocessed_input.value] = (
                    self.prompt_format["prompt_input"].format_map(raw_example).strip()
                )
            else:
                result_example[DatasetKeys.preprocessed_input.value] = (
                    self.prompt_format["prompt_no_input"].format_map(raw_example).strip()
                )

            result_example[DatasetKeys.preprocessed_output.value] = raw_example["output"].strip()

            if not self.is_encoder_decoder:
                result_example[DatasetKeys.preprocessed_output.value] = (
                    " " + result_example[DatasetKeys.preprocessed_output.value]
                )

            if DatasetKeys.id.value not in raw_example:
                result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

            result_example.update(raw_example)
            examples.append(result_example)

        return examples
