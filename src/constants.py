from enum import Enum


DUMMY = "<DUMMY>"


class DatasetSplit(str, Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class Mode(str, Enum):
    """training / inference mode"""

    training = "training"
    inference = "inference"


class OptimizerKeys(str, Enum):
    """standard keys in the optimizer"""

    optimizer_class = "optimizer_class"


class LearningRateScheduler(str, Enum):
    """learning rate schedule"""

    linear = "linear"
    cosine = "cosine"


class DatasetConfigKeys(str, Enum):
    """standard keys in the dataset"""

    data_class = "data_class"
    data_name = "data_name"
    data_path = "data_path"
    data_sampling_proportion = "data_sampling_proportion"
    max_input_tokens = "max_input_tokens"
    max_output_tokens = "max_output_tokens"
    input_format = "input_format"
    output_format = "output_format"


class DatasetKeys(str, Enum):
    """standard keys in the dataset"""

    id = "id"
    input = "input"
    output = "output"
    preprocessed_input = "preprocessed_input"
    preprocessed_output = "preprocessed_output"
    generated_text = "generated_text"
    num_generated_tokens = "num_generated_tokens"
    data_class_index = "data_class_index"


class TrainingInferenceType(str, Enum):
    """training method"""

    full_finetuning = "full_finetuning"
    prompt_tuning = "prompt_tuning"


class PaddingSide(str, Enum):
    """padding side for tokenizer"""

    left = "left"
    right = "right"
