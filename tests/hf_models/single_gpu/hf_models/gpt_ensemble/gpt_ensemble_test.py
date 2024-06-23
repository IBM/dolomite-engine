from dolomite_engine.hf_models import AttentionHeadType, GPTEnsembleConfig, PositionEmbeddingType

from ..gpt_dolomite_test import GPTDolomiteAttentionTest


SEED = 1234


class AttentionTestForDenseModel(GPTDolomiteAttentionTest):
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
            n_embd=32,
            n_layer=num_layers,
            n_head=4,
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
