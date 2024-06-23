import torch
import torch.nn.functional as F
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models.models.gpt_ensemble.linear import EnsembleLinear

from ....test_common import TestCommons


SEED = 1234


class EnsembleLinearTest(TestCommons):
    @parameterized.expand([False, True])
    def test_ensemble_linear_correctness(self, bias: bool) -> None:
        set_seed(SEED)

        tp_world_size = 8
        in_features = 7
        out_features = 11

        ensemble_linear = EnsembleLinear(in_features, out_features, tp_world_size, std=0.02, bias=bias)
        input = torch.randn(4, tp_world_size, 37, in_features)

        output = []
        for rank in range(tp_world_size):
            i = input[:, rank, :, :]
            w = ensemble_linear.weight[rank, :, :].T
            b = None if ensemble_linear.bias is None else ensemble_linear.bias[rank, :]

            output.append(F.linear(i, w, b))

        output = torch.stack(output, dim=1)
        ensemble_output = ensemble_linear(input)

        assert output.equal(ensemble_output)
