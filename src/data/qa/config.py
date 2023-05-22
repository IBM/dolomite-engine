from enum import Enum

from src.constants import DatasetSplit
from src.data.config import DatasetConfig


class ELI5Config(DatasetConfig):
    # max tokens to use for document
    max_document_length: int = 924
    # files for the dataset
    files: dict = {
        DatasetSplit.train.value: "pos_train_ans_retrieved.json",
        DatasetSplit.val.value: "pos_dev_ans_retrieved.json",
        DatasetSplit.test.value: "pos_dev_ans_retrieved.json",
    }
