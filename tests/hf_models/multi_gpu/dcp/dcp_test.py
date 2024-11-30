import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType

from ...test_common import TestCommons


class DCPTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_attention_head_types(), ["gelu", "geglu"], [False, True], [(3, 2, 2), (3, 1, 4), (0, 4, 1)]
        )
        + TestCommons.make_args_matrix(
            [AttentionHeadType.gqa], ["gelu", "geglu"], [False], [(3, 2, 2), (3, 1, 4), (0, 4, 1)]
        )
    )
    @TestCommons.slow_test
    def test_dcp(
        self,
        attention_head_type: AttentionHeadType,
        activation_function: str,
        tensor_parallel_word_embeddings: bool,
        zero_stage_ddp_sizes: tuple[int, int, int],
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.dcp.dcp",
                "--train-config",
                "tests/hf_models/multi_gpu/dcp/train.yml",
                "--unshard-config",
                "tests/hf_models/multi_gpu/dcp/unshard.yml",
                "--attention-head-type",
                attention_head_type.value,
                "--activation-function",
                activation_function,
                "--tmp-path",
                tmp_path,
                "--zero-stage",
                str(zero_stage_ddp_sizes[0]),
                "--data-parallel-replication-world-size",
                str(zero_stage_ddp_sizes[1]),
                "--data-parallel-sharding-world-size",
                str(zero_stage_ddp_sizes[2]),
            ]

            if tensor_parallel_word_embeddings:
                command.append("--tensor-parallel-word-embeddings")

            subprocess.run(command, check=True)
