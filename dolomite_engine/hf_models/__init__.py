from .enums import AttentionHeadType, PositionEmbeddingType
from .modeling_utils_TP import (
    CUDA_RNGStatesTracker,
    get_tensor_parallel_group_manager,
    set_cuda_rng_tracker,
    set_tensor_parallel_group_manager,
)
from .models import (
    DenseMoEConfig,
    DenseMoEForCausalLM,
    DenseMoEModel,
    GPTDolomiteConfig,
    GPTDolomiteForCausalLM,
    GPTDolomiteForCausalLM_TP,
    GPTDolomiteModel,
    GPTDolomiteModel_TP,
    GPTMultiLayerConfig,
    GPTMultiLayerForCausalLM,
    GPTMultiLayerModel,
    MoEDolomiteConfig,
    MoEDolomiteForCausalLM,
    MoEDolomiteModel,
    convert_gpt_dolomite_to_gpt_multilayer,
    export_to_huggingface_bigcode,
    export_to_huggingface_llama,
    export_to_huggingface_mixtral,
    import_from_huggingface_bigcode,
    import_from_huggingface_llama,
    import_from_huggingface_mixtral,
)
from .register_hf import is_custom_model, register_model_classes


register_model_classes()
