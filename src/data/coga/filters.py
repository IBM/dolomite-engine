from functools import partial
from typing import Callable, List, Union

from src.data.config import DatasetConfig


class _Filters:
    @classmethod
    def last_speaker_is_not_agent(cls, data_config, raw_example):
        return data_config.filter_allowed and raw_example["last_speaker"].lower() != "agent"

    @classmethod
    def is_not_original(cls, data_config, raw_example):
        return data_config.filter_allowed and raw_example["neg_subtype"].lower() != "original"

    @classmethod
    def is_data_type_allowed(cls, data_config, raw_example):
        return data_config.allowed_data_type is None or data_config.allowed_data_type in raw_example["data_types"]

    @classmethod
    def is_response_long(cls, data_config, raw_example):
        return (
            data_config.length_threshold is None or len(raw_example["response"].split()) > data_config.length_threshold
        )

    @classmethod
    def is_da_type_allowed(cls, data_config, raw_example):
        return data_config.allowed_da_types is None or raw_example["da"] in data_config.allowed_da_types

    @classmethod
    def is_bertscore_high(cls, data_config, raw_example):
        return (
            data_config.bertscore_threshold is None
            or max(raw_example["bertscore_p"]) > data_config.bertscore_threshold
        )

    @classmethod
    def is_bertscore_low(cls, data_config, raw_example):
        return (
            data_config.bertscore_threshold is None
            or max(raw_example["bertscore_p"]) < data_config.bertscore_threshold
        )

    @classmethod
    def is_unanswerable(cls, data_config, raw_example):
        return raw_example["type"] == "negative"

    @classmethod
    def is_grounded(cls, data_config, raw_example):
        return raw_example["type"] == "positive"


def filter_dataset(data_config: DatasetConfig, examples: List[dict]) -> List[dict]:
    """filters the examples based on the filter_functions in data_config

    Args:
        data_config (DatasetConfig): dataset config
        examples (List[dict]): list of examples

    Returns:
        List[dict]: filtered list of examples
    """

    filter_functions: List[Callable[[dict], bool]] = data_config.filter_functions
    for filter_function_name in filter_functions:
        filter_function = getattr(_Filters, filter_function_name)
        assert filter_function is not None and callable(filter_function)
        examples = list(filter(partial(filter_function, data_config), examples))
    return examples
