import logging
from typing import Union

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode, TuningMethod
from ..utils import log_rank_0, run_rank_n
from .base import ModelWrapper
from .finetuning import ModelWrapperForFinetuning
from .peft import ModelWrapperForPEFT
from .pretraining import ModelWrapperForPretraining


_MODEL_CLASS_MAPPING = {
    TuningMethod.pretraining: ModelWrapperForPretraining,
    TuningMethod.full_finetuning: ModelWrapperForFinetuning,
    TuningMethod.lora: ModelWrapperForPEFT,
    TuningMethod.prompt_tuning: ModelWrapperForPEFT,
}


def get_model(args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode) -> ModelWrapper:
    tuning_method = args.tuning_args.tuning_method
    if tuning_method in _MODEL_CLASS_MAPPING:
        return _MODEL_CLASS_MAPPING[tuning_method](args, mode)

    raise ValueError(f"unexpected tuning_method ({tuning_method})")


@run_rank_n
def log_model(model: ModelWrapper) -> None:
    """print model

    Args:
        model (ModelWrapper): model to print
    """

    log_rank_0(logging.INFO, "------------------------ model ------------------------")
    log_rank_0(logging.INFO, model)
    log_rank_0(logging.INFO, "-------------------- end of model ---------------------")
