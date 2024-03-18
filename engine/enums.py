from enum import Enum


class AttentionImplementation(Enum):
    """
    Enum class for attention implementation
    """

    eager = "eager"
    sdpa = "sdpa"
    flash_attention_2 = "flash_attention_2"


class DatasetSplit(str, Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class Mode(str, Enum):
    """training / inference mode"""

    training = "training"
    inference = "inference"
    export = "export"


class PaddingSide(str, Enum):
    """padding side for the tokenizer"""

    left = "left"
    right = "right"


class DatasetKeys(str, Enum):
    """standard keys in the dataset"""

    input = "input"
    output = "output"
    generated_text = "generated_text"
    num_generated_tokens = "num_generated_tokens"


class TuningMethod(str, Enum):
    """training method"""

    pretraining = "pretraining"
    full_finetuning = "full_finetuning"
    prompt_tuning = "prompt_tuning"
    lora = "lora"


class DistributedBackend(str, Enum):
    deepspeed = "deepspeed"
    torch = "torch"


class ArgsFileExtension(str, Enum):
    json = "json"
    yaml = "yaml"


class LossMask(str, Enum):
    """Type of loss masking method"""

    output_only = "output_only"
    no_mask = "no_mask"
