from typing import Any, Mapping

import torch.nn as nn

from .TP import modify_state_dict_to_dtensor_dict


class DTensorModule(nn.Module):
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)
