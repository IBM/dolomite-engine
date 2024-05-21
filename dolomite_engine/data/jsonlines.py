import glob
import os
from typing import List

from transformers import AutoTokenizer

from ..enums import DatasetKeys, DatasetSplit, Mode, TuningMethod
from .base import BaseDataset


class JSONLinesDataset(BaseDataset):
    """A dataset for loading JSON lines files"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        tuning_method: TuningMethod,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
        num_virtual_tokens: int = None,
    ) -> None:
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            tuning_method=tuning_method,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_virtual_tokens=num_virtual_tokens,
        )

        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        import jsonlines

        assert "data_path" in self.class_args, "JSONLinesDataset requires additional class_args `data_path`"

        examples = []
        data_files = glob.glob(os.path.join(self.class_args["data_path"], self.split.value, "*.jsonl"))

        for filename in data_files:
            json_file = jsonlines.open(filename, "r")

            for raw_example in json_file:
                input = self.construct_input_from_format(raw_example[DatasetKeys.input.value])
                output = (
                    self.construct_output_from_format(raw_example[DatasetKeys.output.value])
                    if self.mode == Mode.training
                    else None
                )

                example = self.get_input_output_token_ids(input, output)
                examples.append(example)

        return examples
