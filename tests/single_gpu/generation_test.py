import torch
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, DenseMoEConfig, PositionEmbeddingType
from dolomite_engine.hf_models.models.bigcode import _export_config_to_huggingface

from ..test_common import TestCommons


class GenerationTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            [AttentionHeadType.mha, AttentionHeadType.mqa],
            [PositionEmbeddingType.learned_absolute],
            TestCommons.get_dtypes(),
            [True, False],
        )
    )
    def test_generation_matches_bigcode(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
        use_cache: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        megatron_config = self.get_dense_test_config(attention_head_type, position_embedding_type)
        hf_config = _export_config_to_huggingface(megatron_config)

        megatron_config.use_cache = use_cache
        hf_config.use_cache = use_cache

        megatron_model = self.from_config(megatron_config, torch_dtype=torch_dtype).to(device)
        megatron_model.eval()

        hf_model = self.from_config(hf_config, torch_dtype=torch_dtype).to(device)
        hf_model.load_state_dict(megatron_model.state_dict())
        hf_model.eval()

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        megatron_output = megatron_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, return_dict_in_generate=True, output_scores=True
        )
        hf_output = hf_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, return_dict_in_generate=True, output_scores=True
        )

        assert megatron_output.sequences.equal(hf_output.sequences)

        for token_score_without_cache, token_score_with_cache in zip(megatron_output.scores, hf_output.scores):
            self.assert_equal_tensors(
                token_score_without_cache,
                token_score_with_cache,
                False,
                rtol_float32=0,
                atol_float32=3e-7,
                rtol_float16=1e-2,
                atol_float16=4e-4,
                rtol_bfloat16=5e-3,
                atol_bfloat16=5e-3,
            )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            TestCommons.get_dtypes(),
        )
    )
    def test_generation_works(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        for config in [
            self.get_dense_test_config(attention_head_type, position_embedding_type),
            self.get_moe_test_config(attention_head_type, position_embedding_type),
        ]:
            model = self.from_config(config, torch_dtype=torch_dtype).to(device)
            model.eval()

            input_ids, attention_mask, _ = self.get_dummy_inputs(device)

            model.generate(input_ids=input_ids, attention_mask=attention_mask)

    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_dtypes()))
    def test_generation_works_dense_moe(self, device: torch.device, torch_dtype: torch.dtype) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        original_config = self.get_dense_test_config(AttentionHeadType.gqa, PositionEmbeddingType.learned_absolute)

        config = DenseMoEConfig(
            vocab_size=original_config.vocab_size,
            n_positions=original_config.n_positions,
            n_embd=original_config.n_embd,
            n_layer=original_config.n_layer,
            n_head=original_config.n_head,
            num_experts=original_config.num_key_value_heads,
            attention_head_type=original_config.attention_head_type,
            position_embedding_type=original_config.position_embedding_type,
            add_bias=original_config.add_bias,
            activation_function=original_config.activation_function,
            normalization_function=original_config.normalization_function,
            tie_word_embeddings=original_config.tie_word_embeddings,
        )
        model = self.from_config(config, torch_dtype=torch_dtype).to(device)
        model.eval()

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        model.generate(input_ids=input_ids, attention_mask=attention_mask)
