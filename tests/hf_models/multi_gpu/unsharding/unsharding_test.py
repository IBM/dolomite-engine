import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, GPTDolomiteConfig, MoEDolomiteConfig

from ...test_common import TestCommons


class UnshardingTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_attention_head_types(),
            ["gelu", "geglu"],
            [False, True],
            [GPTDolomiteConfig.model_type],
        )
        + TestCommons.make_args_matrix(
            [AttentionHeadType.gqa],
            ["gelu", "geglu"],
            [False],
            [MoEDolomiteConfig.model_type],
        )
    )
    @TestCommons.slow_test
    def test_unsharding(
        self,
        attention_head_type: AttentionHeadType,
        activation_function: str,
        tensor_parallel_word_embeddings: bool,
        model_type: str,
    ) -> None:
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
                "--model-type",
                model_type,
            ]

            if tensor_parallel_word_embeddings:
                command.append("--tensor-parallel-word-embeddings")

            subprocess.run(command, check=True)
