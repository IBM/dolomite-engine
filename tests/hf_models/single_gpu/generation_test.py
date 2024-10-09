import os
import tempfile

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType, export_to_huggingface

from ..test_common import TestCommons


class GenerationTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            [AttentionHeadType.mha, AttentionHeadType.mqa],
            [PositionEmbeddingType.learned_absolute],
            [torch.float32],
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

        dolomite_config = self.get_dense_test_config(attention_head_type, position_embedding_type)
        dolomite_config.use_cache = use_cache

        dolomite_model = self.from_config(dolomite_config, torch_dtype=torch_dtype).to(device)
        dolomite_model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            dolomite_path = os.path.join(tmpdir, "dolomite")
            bigcode_path = os.path.join(tmpdir, "bigcode")

            dolomite_model.save_pretrained(dolomite_path)
            export_to_huggingface(dolomite_path, bigcode_path, model_type="gpt_bigcode")

            bigcode_model = AutoModelForCausalLM.from_pretrained(bigcode_path).to(device)
            bigcode_model.eval()

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        dolomite_output = dolomite_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )
        bigcode_output = bigcode_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert dolomite_output.sequences.equal(bigcode_output.sequences)

        for dolomite_token_score, bigcode_token_score in zip(dolomite_output.scores, bigcode_output.scores):
            self.assert_equal_tensors(
                dolomite_token_score,
                bigcode_token_score,
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
