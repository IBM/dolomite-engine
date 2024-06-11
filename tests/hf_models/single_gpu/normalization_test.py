import torch
from parameterized import parameterized

from dolomite_engine.hf_models.modeling_utils import get_normalization_function
from dolomite_engine.utils import is_apex_available

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


class NormTest(TestCommons):
    def _skip_test_if_apex_unavailable(self) -> None:
        if not is_apex_available():
            self.skipTest("skipping test because apex is not installed")

    def _test_equivalance(
        self, device: torch.device, hidden_size: int, normalization_function: str, normalization_implementation: str
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        x = torch.randn(100, hidden_size, device=device)

        torch_norm = get_normalization_function(
            normalization_function, hidden_size, normalization_implementation="torch"
        ).to(device)
        torch_output = torch_norm(x)

        custom_norm = get_normalization_function(
            normalization_function, hidden_size, normalization_implementation=normalization_implementation
        ).to(device)
        custom_output = custom_norm(x)

        self.assert_equal_tensors(torch_output, custom_output, False)

    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_layernorm_apex_layernorm_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self._skip_test_if_apex_unavailable()
        self._test_equivalance(device, hidden_size, "layernorm", "apex")

    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_layernorm_apex_persistent_layernorm_equivalence(
        self, device: torch.device, hidden_size: int
    ) -> None:
        self._skip_test_if_apex_unavailable()
        self._test_equivalance(device, hidden_size, "layernorm", "apex_persistent")

    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], TEST_HIDDEN_SIZES))
    def test_torch_rmsnorm_apex_rmsnorm_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self._skip_test_if_apex_unavailable()
        self._test_equivalance(device, hidden_size, "rmsnorm", "apex")

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")], [1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192]
        )
    )
    def test_torch_rmsnorm_torchtitan_rmsnorm_equivalence(self, device: torch.device, hidden_size: int) -> None:
        self._test_equivalance(device, hidden_size, "rmsnorm", "torchtitan")
