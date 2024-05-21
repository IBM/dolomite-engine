import os
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType

from ..test_common import TestCommons


# class TensorParallelTest(TestCommons):
#     @parameterized.expand(
#         TestCommons.make_args_matrix(
#             TestCommons.get_attention_head_types(),
#             TestCommons.get_position_embedding_types(),
#             TestCommons.get_attention_implementations(),
#         )
#     )
#     def test_tensor_parallel_forward(
#         self,
#         attention_head_type: AttentionHeadType,
#         position_embedding_type: PositionEmbeddingType,
#         attention_implementation: str,
#     ) -> None:
#         self.skip_test_if_device_unavailable(torch.device("cuda"))
#         if attention_implementation == "flash_attention_2" and position_embedding_type == PositionEmbeddingType.alibi:
#             self.skipTest("skipping test because Alibi is not supported with flash attention")

#         gpus_per_node = torch.cuda.device_count()

#         with tempfile.TemporaryDirectory() as tmp_path:
#             outfile = os.path.join(tmp_path, "out.log")

#             command = (
#                 f"torchrun --nproc_per_node {gpus_per_node} -m tests.multi_gpu.tensor_parallel_forward "
#                 f"--attention-head-type {attention_head_type.value} "
#                 f"--position-embedding-type {position_embedding_type.value} "
#                 f"--attention-implementation {attention_implementation} "
#                 f"--tmp-path {tmp_path} |& tee {outfile}"
#             )
#             os.system(command)

#             last_line = open(outfile, "r").readlines()[-1].strip()

#         error = last_line.lstrip("tensor(").rsplit(",")[0]
#         error = float(error)

#         assert error == 0, "outputs don't match for normal and tensor parallel model"
