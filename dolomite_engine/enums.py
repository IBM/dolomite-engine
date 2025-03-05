from enum import Enum


class ParamsGroupMethod(Enum):
    mup = "mup"


class GradientCheckpointingMethod(Enum):
    block = "block"


class LRDecaySchedule(Enum):
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


class DatasetSplit(Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class Mode(Enum):
    """training / inference mode"""

    training = "training"
    inference = "inference"
    unsharding = "unsharding"
    distillation = "distillation"


class TuningMethod(Enum):
    """training method"""

    pretraining = "pretraining"
    full_finetuning = "full_finetuning"
    distillation = "distillation"


class LossMask(Enum):
    """Type of loss masking method"""

    output_only = "output_only"
    no_mask = "no_mask"


class KLDivergenceMethod(Enum):
    """Type of KL divergence"""

    forward = "forward"
    backward = "backward"


class ExperimentsTrackerName(Enum):
    """Experiment tracker to use"""

    aim = "aim"
    wandb = "wandb"


class Kernel(Enum):
    rmsnorm_cute = "rmsnorm_cute"
    swiglu_unchunked_cute = "swiglu_unchunked_cute"
    mamba2_ssm = "mamba2_ssm"
    scattermoe = "scattermoe"
    cross_entropy_cute = "cross_entropy_cute"
    fused_linear_cross_entropy_cute = "fused_linear_cross_entropy_cute"
