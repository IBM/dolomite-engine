import torch
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ..test_common import TestCommons


SEED = 1234


class AttentionSupportedTest(TestCommons):
    @parameterized.expand(TestCommons.get_attention_implementations())
    def test_attention_implementation_supported_dense(self, attention_implementation: str) -> None:
        if attention_implementation == "flash_attention_2":
            self.skip_test_if_device_unavailable(torch.device("cuda"))

        self.from_config(
            self.get_dense_test_config(AttentionHeadType.mha, PositionEmbeddingType.learned_absolute),
            attn_implementation=attention_implementation,
        )

    @parameterized.expand(TestCommons.get_attention_implementations())
    def test_attention_implementation_supported_moe(self, attention_implementation: str) -> None:
        if attention_implementation == "flash_attention_2":
            self.skip_test_if_device_unavailable(torch.device("cuda"))

        self.from_config(
            self.get_moe_test_config(AttentionHeadType.mha, PositionEmbeddingType.learned_absolute),
            attn_implementation=attention_implementation,
        )
