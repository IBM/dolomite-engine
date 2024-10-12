from enum import Enum


class ParamsGroupMethod(Enum):
    mup = "mup"


class GradientCheckpointingMethod(Enum):
    block = "block"


class LRDecaySchedule(str, Enum):
    constant = "constant"
    cosine = "cosine"
    exponential = "exponential"
    linear = "linear"
    power = "power"


class AttentionImplementation(Enum):
    """
    Enum class for attention implementation
    """

    eager = "eager"
    sdpa = "sdpa"
    flash_attention_2 = "flash_attention_2"


class MoEImplementation(Enum):
    """
    Enum class for MoE implementation
    """

    eager = "eager"
    scattermoe = "scattermoe"


class DatasetSplit(str, Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class Mode(str, Enum):
    """training / inference mode"""

    training = "training"
    inference = "inference"
    unsharding = "unsharding"
    distillation = "distillation"


class TuningMethod(str, Enum):
    """training method"""

    pretraining = "pretraining"
    full_finetuning = "full_finetuning"
    prompt_tuning = "prompt_tuning"
    lora = "lora"
    distillation = "distillation"


class FP8Backend(str, Enum):
    msamp = "msamp"
    nvte = "nvte"


class LossMask(str, Enum):
    """Type of loss masking method"""

    output_only = "output_only"
    no_mask = "no_mask"


class KLDivergenceMethod(str, Enum):
    """Type of KL divergence"""

    forward = "forward"
    backward = "backward"


class ExperimentsTrackerName(str, Enum):
    """Experiment tracker to use"""

    aim = "aim"
    wandb = "wandb"
