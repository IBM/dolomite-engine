import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType
from dolomite_engine.utils import torch_dtype_to_string

from ...test_common import TestCommons


class TensorParallelTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            ["gpt_dolomite"],
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            TestCommons.get_attention_implementations(),
            TestCommons.get_dtypes(),
            [False, True],
            [False, True],
        )
        + TestCommons.make_args_matrix(
            ["gpt_ensemble"],
            [AttentionHeadType.mha, AttentionHeadType.gqa],
            [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.rope],
            ["sdpa"],
            [torch.float32],
            [False],
            [False],
        )
    )
    def test_tensor_parallel_forward(
        self,
        model_type: str,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        attention_implementation: str,
        torch_dtype: torch.dtype,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))
        if attention_implementation == "flash_attention_2" and position_embedding_type == PositionEmbeddingType.alibi:
            self.skipTest("skipping test because Alibi is not supported with flash attention")

        if (attention_implementation, torch_dtype) not in [
            ("eager", torch.float32),
            ("sdpa", torch.float32),
            ("flash_attention_2", torch.float16),
        ]:
            self.skipTest("skipping test since running all takes too long")

        if use_padding_free_transformer and attention_implementation != "flash_attention_2":
            print(use_padding_free_transformer, attention_implementation)
            self.skipTest("skipping test since flash attention is needed for padding free transformer")

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.tensor_parallel.forward",
                "--attention-head-type",
                attention_head_type.value,
                "--position-embedding-type",
                position_embedding_type.value,
                "--torch-dtype",
                torch_dtype_to_string(torch_dtype),
                "--attention-implementation",
                attention_implementation,
                "--tmp-path",
                tmp_path,
                "--model-type",
                model_type,
            ]

            if use_padding_free_transformer:
                command.append("--use-padding-free-transformer")

            if sequence_parallel:
                command.append("--sequence-parallel")

            subprocess.run(command, check=True)