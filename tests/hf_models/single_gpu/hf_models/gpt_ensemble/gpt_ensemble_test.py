import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models import AttentionHeadType, GPTEnsembleConfig, PositionEmbeddingType

from ....test_common import TestCommons


SEED = 1234


class AttentionTestForEnsembleModel(TestCommons):
    @staticmethod
    def get_dense_test_config(
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        num_layers: int = 8,
        add_bias: bool = True,
        activation_function: str = "gelu_pytorch_tanh",
        normalization_function: str = "rmsnorm",
    ) -> GPTEnsembleConfig:
        return GPTEnsembleConfig(
            vocab_size=2048,
            n_positions=1024,
            n_embd=96,
            n_layer=num_layers,
            n_head=12,
            num_key_value_heads=4 if attention_head_type == AttentionHeadType.gqa else None,
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
            [AttentionHeadType.mha, AttentionHeadType.gqa],
            [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.rope],
            [torch.float32, torch.float16],
        )
    )
    def test_math_attention_sdpa_equivalence(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        set_seed(SEED)

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        config = self.get_dense_test_config(attention_head_type, position_embedding_type)

        math_model = self.from_config(config, torch_dtype=torch_dtype, attn_implementation="eager").to(device)
        sdpa_model = self.from_config(config, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)

        math_model.eval()
        sdpa_model.eval()

        sdpa_model.load_state_dict(math_model.state_dict())

        math_output = math_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        math_logits = math_output.logits
        math_loss = math_output.loss

        sdpa_output = sdpa_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        sdpa_logits = sdpa_output.logits
        sdpa_loss = sdpa_output.loss

        # we don't care about what happens on masked values (they don't match btw)
        math_logits[attention_mask == 0] = 0
        sdpa_logits[attention_mask == 0] = 0

        self.assert_equal_tensors(
            math_logits,
            sdpa_logits,
            False,
            rtol_float32=0,
            atol_float32=6e-7,
            rtol_float16=1e-2,
            atol_float16=7e-4,
            rtol_bfloat16=5e-3,
            atol_bfloat16=5e-3,
        )
        self.assert_equal_tensors(
            math_loss,
            sdpa_loss,
            False,
            rtol_float32=0,
            atol_float32=1e-5,
            rtol_float16=0,
            atol_float16=1e-5,
            rtol_bfloat16=0,
            atol_bfloat16=1e-5,
        )
