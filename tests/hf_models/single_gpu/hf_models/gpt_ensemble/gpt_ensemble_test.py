import torch
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, GPTEnsembleConfig, PositionEmbeddingType

from ....test_common import TestCommons
from ..gpt_dolomite_test import GPTDolomiteAttentionTest


class AttentionTestForDenseModel(TestCommons):
    @staticmethod
    def get_dense_test_config(
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        num_layers: int = 8,
        add_bias: bool = True,
        activation_function: str = "gelu_pytorch_tanh",
        normalization_function: str = "layernorm",
    ) -> GPTEnsembleConfig:
        return GPTEnsembleConfig(
            vocab_size=2048,
            n_positions=1024,
            n_embd=96,
            n_layer=num_layers,
            n_head=12,
            num_key_value_heads=2 if attention_head_type == AttentionHeadType.gqa else None,
            attention_head_type=attention_head_type.value,
            position_embedding_type=position_embedding_type.value,
            add_bias=add_bias,
            activation_function=activation_function,
            normalization_function=normalization_function,
            tie_word_embeddings=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            pretraining_tensor_parallel_size=4,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types()[:1],
            [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.rope],
            TestCommons.get_dtypes(),
        )
    )
    def test_math_attention_sdpa_equivalence(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
    ) -> None:
        GPTDolomiteAttentionTest._test_math_attention_sdpa_equivalence(
            self,
            device=device,
            attention_head_type=attention_head_type,
            position_embedding_type=position_embedding_type,
            torch_dtype=torch_dtype,
        )
