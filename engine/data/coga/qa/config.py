from enum import Enum
from typing import List

from engine.constants import DatasetSplit
from engine.data.config import DatasetConfig


class ELI5Config(DatasetConfig):
    # max tokens to use for document
    max_document_length: int = 924
    # whether to use retrieved passages
    use_retrieved_passages: bool = False
    # threshold for matching passage to response
    bertscore_threshold: float = None
    # filter by allowed data types
    allowed_da_types: List[str] = None
    # filter functions to use
    filter_functions: List[str] = []
    # select only examples with response length greater than this
    length_threshold: int = None
    # whether to use the control prompt
    control_prompt: str = None

    # files for the dataset
    files: dict = {
        DatasetSplit.train.value: "pos_train_ans_retrieved.json",
        DatasetSplit.val.value: "pos_dev_ans_retrieved.json",
        DatasetSplit.test.value: "pos_dev_ans_retrieved.json",
    }
