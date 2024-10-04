from ..dense_TP import CausalLMModelMixin_TP
from ..moe import CausalLMMoEModelMixin


class CausalLMMoEModelMixin_TP(CausalLMMoEModelMixin, CausalLMModelMixin_TP): ...
