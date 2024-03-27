from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ...test_common import TestCommons


class AttentionImplementationTest(TestCommons):
    def test_attention_implementation_works(self) -> None:
        config = self.get_dense_test_config(AttentionHeadType.mha, PositionEmbeddingType.learned_absolute)
        model = self.from_config(config, attn_implementation="eager")

        model_repr = str(model)
        assert "MathAttention" in model_repr
