from ..gpt_dolomite_TP import GPTDolomiteModel_TP, GPTDolomitePreTrainedModel_TP
from ..gpt_ensemble import GPTEnsembleConfig
from .layer import GPTEnsembleBlock_TP


class GPTEnsemblePreTrainedModel_TP(GPTDolomitePreTrainedModel_TP):
    config_class = GPTEnsembleConfig
    layer_class = GPTEnsembleBlock_TP
    _no_split_modules = ["GPTEnsembleBlock_TP"]


class GPTEnsembleModel_TP(GPTEnsemblePreTrainedModel_TP, GPTDolomiteModel_TP): ...
