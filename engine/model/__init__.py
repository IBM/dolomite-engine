import logging
from typing import Union

from ..arguments import ExportArgs, InferenceArgs, TrainingArgs
from ..enums import Mode, TuningMethod
from ..utils import log_rank_0, run_rank_n
from .base import Model
from .finetuning import ModelForFinetuning
from .peft import ModelForPEFT
from .pretraining import ModelForPretraining


_MODEL_CLASS_MAPPING = {
    TuningMethod.pretraining: ModelForPretraining,
    TuningMethod.full_finetuning: ModelForFinetuning,
    TuningMethod.lora: ModelForPEFT,
    TuningMethod.prompt_tuning: ModelForPEFT,
}


def get_model(args: Union[TrainingArgs, InferenceArgs, ExportArgs], mode: Mode) -> Model:
    tuning_method = args.tuning_args.tuning_method
    if tuning_method in _MODEL_CLASS_MAPPING:
        return _MODEL_CLASS_MAPPING[tuning_method](args, mode)

    raise ValueError(f"unexpected tuning_method ({tuning_method})")


@run_rank_n
def log_model(model: Model) -> None:
    """print model

    Args:
        model (Model): model to print
    """

    log_rank_0(logging.INFO, "------------------------ model ------------------------")
    log_rank_0(logging.INFO, model)
    log_rank_0(logging.INFO, "-------------------- end of model ---------------------")
