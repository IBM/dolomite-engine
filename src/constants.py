from enum import Enum


DUMMY = "<DUMMY>"


class DatasetSplit(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class Mode(str, Enum):
    training = "training"
    inference = "inference"


class LearningRateScheduler(str, Enum):
    linear = "linear"
    cosine = "cosine"


class DatasetKeys(str, Enum):
    id = "id"
    input = "input"
    output = "output"
    preprocessed_input = "preprocessed_input"
    preprocessed_output = "preprocessed_output"
    generated_text = "generated_text"
    data_class_index = "data_class_index"


class TrainingInferenceType(str, Enum):
    full_finetuning = "full_finetuning"
    prompt_tuning = "prompt_tuning"
