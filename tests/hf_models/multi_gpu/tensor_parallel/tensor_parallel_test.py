import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ...test_common import TestCommons


class TensorParallelTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            ["eager", "sdpa"],
        )
    )
    def test_tensor_parallel_forward(
        self,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        attention_implementation: str,
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))
        if attention_implementation == "flash_attention_2" and position_embedding_type == PositionEmbeddingType.alibi:
            self.skipTest("skipping test because Alibi is not supported with flash attention")

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.tensor_parallel.tensor_parallel_forward",
                "--attention-head-type",
                str(attention_head_type.value),
                "--position-embedding-type",
                str(position_embedding_type.value),
                "--attention-implementation",
                str(attention_implementation),
                "--tmp-path",
                str(tmp_path),
            ]

            subprocess.run(command, check=True)
