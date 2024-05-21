import torch
from parameterized import parameterized

from dolomite_engine.hf_models import GPTDolomiteConfig

from ..test_common import TestCommons


class TypeCheckTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")]))
    def test_no_attention_mask_flash_attention(self, device: torch.device) -> None:
        self.skip_test_if_device_unavailable(device)

        config = GPTDolomiteConfig(2048, 1024, 32, 8, 4)
        model = self.from_config(
            config, attn_implementation="flash_attention_2", use_padding_free_transformer=True
        ).to(device)
        model.eval()

        input_ids, _, labels = self.get_dummy_inputs(device, return_list=True)
        attention_mask = [[1] * len(i) for i in input_ids]

        self.assertRaises(AssertionError, model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
