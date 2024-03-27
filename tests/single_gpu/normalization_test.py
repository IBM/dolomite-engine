import torch
from parameterized import parameterized

from dolomite_engine.hf_models.modeling_utils import get_normalization_function

from ..test_common import TestCommons


TEST_HIDDEN_SIZES = [
    1024,
    1536,
    2048,
    2304,
    3072,
    3840,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    12800,
    15360,
    16384,
    18432,
    20480,
    24576,
    25600,
    30720,
    32768,
    40960,
    49152,
    65536,
]


class LayerNormTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_apex_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self.skip_test_if_device_unavailable(device)

        x = torch.randn(100, hidden_size, device=device)

        torch_layernorm = get_normalization_function(
            "layernorm", hidden_size, normalization_implementation="torch"
        ).to(device)
        torch_output = torch_layernorm(x)

        apex_layernorm = get_normalization_function("layernorm", hidden_size, normalization_implementation="apex").to(
            device
        )
        apex_output = apex_layernorm(x)

        self.assert_equal_tensors(torch_output, apex_output, False)

    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_apex_persistent_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self.skip_test_if_device_unavailable(device)

        x = torch.randn(100, hidden_size, device=device)

        torch_layernorm = get_normalization_function(
            "layernorm", hidden_size, normalization_implementation="torch"
        ).to(device)
        torch_output = torch_layernorm(x)

        apex_layernorm = get_normalization_function(
            "layernorm", hidden_size, normalization_implementation="apex_persistent"
        ).to(device)
        apex_output = apex_layernorm(x)

        self.assert_equal_tensors(torch_output, apex_output, False)


class RMSNormTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_apex_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self.skip_test_if_device_unavailable(device)

        x = torch.randn(100, hidden_size, device=device)

        torch_rmsnorm = get_normalization_function("rmsnorm", hidden_size, normalization_implementation="torch").to(
            device
        )
        torch_output = torch_rmsnorm(x)

        apex_rmsnorm = get_normalization_function("rmsnorm", hidden_size, normalization_implementation="apex").to(
            device
        )
        apex_output = apex_rmsnorm(x)

        self.assert_equal_tensors(torch_output, apex_output, False)
