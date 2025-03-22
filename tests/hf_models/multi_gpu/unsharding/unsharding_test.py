import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from ...test_common import TestCommons


class UnshardingTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_attention_head_types(), ["gelu", "geglu"]))
    @TestCommons.slow_test
    def test_unsharding(self, attention_head_type: str, activation_function: str) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.unsharding.unsharding",
                "--attention-head-type",
                attention_head_type.value,
                "--activation-function",
                activation_function,
                "--tmp-path",
                tmp_path,
            ]

            subprocess.run(command, check=True)
