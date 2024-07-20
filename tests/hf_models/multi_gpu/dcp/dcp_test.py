import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType

from ...test_common import TestCommons


# class DCPTest(TestCommons):
#     @parameterized.expand(
#         TestCommons.make_args_matrix(TestCommons.get_attention_head_types(), ["gelu", "geglu"], [False, True])
#     )
#     def test_dcp(
#         self, attention_head_type: AttentionHeadType, activation_function: str, tensor_parallel_word_embeddings: bool
#     ) -> None:
#         self.skip_test_if_device_unavailable(torch.device("cuda"))

#         gpus_per_node = torch.cuda.device_count()

#         with tempfile.TemporaryDirectory() as tmp_path:
#             command = [
#                 "torchrun",
#                 "--nproc_per_node",
#                 str(gpus_per_node),
#                 "-m",
#                 "tests.hf_models.multi_gpu.dcp.dcp",
#                 "--train-config",
#                 "tests/hf_models/multi_gpu/dcp/train.yml",
#                 "--unshard-config",
#                 "tests/hf_models/multi_gpu/dcp/unshard.yml",
#                 "--attention-head-type",
#                 attention_head_type.value,
#                 "--activation-function",
#                 activation_function,
#                 "--tmp-path",
#                 tmp_path,
#             ]

#             if tensor_parallel_word_embeddings:
#                 command.append("--tensor-parallel-word-embeddings")

#             subprocess.run(command, check=True)
