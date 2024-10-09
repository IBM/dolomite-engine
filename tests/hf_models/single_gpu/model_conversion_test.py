import torch
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ..test_common import TestCommons


class ModelConversionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), [AttentionHeadType.mha, AttentionHeadType.mqa])
    )
    def test_bigcode_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        dolomite_config = self.get_dense_test_config(attention_head_type, PositionEmbeddingType.learned_absolute)

        self.model_conversion_test(
            dolomite_config=dolomite_config, model_type="gpt_bigcode", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(), TestCommons.get_attention_head_types(), [True, False]
        )
    )
    def test_llama_model_conversion(
        self, device: torch.device, attention_head_type: AttentionHeadType, add_bias: bool
    ) -> None:
        dolomite_config = self.get_dense_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=add_bias,
            activation_function="swiglu",
            normalization_function="rmsnorm",
        )

        self.model_conversion_test(
            dolomite_config=dolomite_config, model_type="llama", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(), TestCommons.get_attention_head_types(), [True, False]
        )
    )
    def test_granite_model_conversion(
        self, device: torch.device, attention_head_type: AttentionHeadType, add_bias: bool
    ) -> None:
        dolomite_config = self.get_dense_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=add_bias,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        self.model_conversion_test(
            dolomite_config=dolomite_config, model_type="granite", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_mixtral_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        dolomite_config = self.get_moe_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
        )

        self.model_conversion_test(
            dolomite_config=dolomite_config,
            model_type="mixtral",
            device=device,
            exact_match=False,
            compare_loss=False,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_granitemoe_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        dolomite_config = self.get_moe_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        self.model_conversion_test(
            dolomite_config=dolomite_config,
            model_type="granitemoe",
            device=device,
            exact_match=False,
            compare_loss=False,
        )
