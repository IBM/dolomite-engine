import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ...test_common import TestCommons


SEED = 1234


class AttentionTestForMoEModel(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
        )
    )
    def test_math_attention_sdpa_equivalence(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        config = self.get_moe_test_config(attention_head_type, position_embedding_type)

        math_model = self.from_config(config, attn_implementation="eager").to(device)
        sdpa_model = self.from_config(config, attn_implementation="sdpa").to(device)

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
            atol_float32=3e-7,
            rtol_float16=0,
            atol_float16=3e-7,
            rtol_bfloat16=0,
            atol_bfloat16=3e-7,
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

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_attention_head_types(),
            [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.rope],
            [torch.float16, torch.bfloat16],
        )
    )
    def test_sdpa_padding_free_transformer_equivalence(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        config = self.get_moe_test_config(attention_head_type, position_embedding_type, num_layers=1)

        sdpa_model = self.from_config(config, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)
        flash_model = self.from_config(
            config,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            use_padding_free_transformer=True,
        ).to(device)

        sdpa_model.eval()
        flash_model.eval()

        flash_model.load_state_dict(sdpa_model.state_dict())

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        sdpa_output = sdpa_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        attention_mask = attention_mask.to(torch.bool)
        sdpa_logits = sdpa_output.logits
        sdpa_logits = torch.cat([sdpa_logits[i, ex, :] for i, ex in enumerate(attention_mask)])
        sdpa_loss = sdpa_output.loss

        input_ids, attention_mask, labels = self.get_dummy_inputs(device, return_list=True)
        flash_output = flash_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        flash_logits = flash_output.logits
        flash_loss = flash_output.loss

        self.assert_equal_tensors(
            sdpa_logits,
            flash_logits,
            False,
            rtol_float16=1e-3,
            atol_float16=3e-4,
            rtol_bfloat16=5e-3,
            atol_bfloat16=5e-3,
        )
        self.assert_equal_tensors(sdpa_loss, flash_loss, False)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_attention_head_types(),
            [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.rope],
            [torch.float16, torch.bfloat16],
        )
    )
    def test_sdpa_flash_attention_equivalence(
        self,
        device: torch.device,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        torch_dtype: torch.dtype,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        config = self.get_moe_test_config(attention_head_type, position_embedding_type, num_layers=1)

        sdpa_model = self.from_config(config, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)
        flash_model = self.from_config(config, torch_dtype=torch_dtype, attn_implementation="flash_attention_2").to(
            device
        )

        sdpa_model.eval()
        flash_model.eval()

        flash_model.load_state_dict(sdpa_model.state_dict())

        sdpa_output = sdpa_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        sdpa_logits = sdpa_output.logits
        sdpa_loss = sdpa_output.loss

        flash_output = flash_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        flash_logits = flash_output.logits
        flash_loss = flash_output.loss

        # we don't care about what happens on masked values (they don't match btw)
        sdpa_logits[attention_mask == 0] = 0
        flash_logits[attention_mask == 0] = 0

        self.assert_equal_tensors(
            sdpa_logits[attention_mask],
            flash_logits[attention_mask],
            False,
            rtol_float16=1e-3,
            atol_float16=3e-4,
            rtol_bfloat16=5e-3,
            atol_bfloat16=5e-3,
        )
        self.assert_equal_tensors(sdpa_loss, flash_loss, False)
