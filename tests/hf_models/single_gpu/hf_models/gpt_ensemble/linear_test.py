import torch
import torch.nn.functional as F
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models.models.gpt_ensemble.linear import Linear3D

from ....test_common import TestCommons


SEED = 1234


class Linear3DTest(TestCommons):
    @parameterized.expand([False, True])
    def test_linear_3d_correctness(self, bias: bool) -> None:
        set_seed(SEED)

        tp_world_size = 8
        in_features = 7
        out_features = 11

        linear3d = Linear3D(in_features, out_features, tp_world_size, std=0.02, bias=bias)
        input = torch.randn(4, 37, tp_world_size, in_features)

        output = []
        for rank in range(tp_world_size):
            i = input[:, :, rank, :]
            w = linear3d.weight[rank, :, :].T
            b = None if linear3d.bias is None else linear3d.bias[rank, :]

            output.append(F.linear(i, w, b))

        output = torch.stack(output, dim=2)
        output3d = linear3d(input)

        assert output.equal(output3d)
