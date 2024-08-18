import torch
import torch.nn as nn
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
        input = torch.randn(tp_world_size, 4, 37, in_features)

        if bias:
            with torch.no_grad():
                nn.init.normal_(ensemble_linear.bias, mean=0, std=ensemble_linear.std)

        output = []
        for rank in range(tp_world_size):
            i = input[rank, ...]
            w = ensemble_linear.weight.view(tp_world_size, in_features, out_features)[rank, :, :].T
            b = None if ensemble_linear.bias is None else ensemble_linear.bias[rank, :]

            output.append(F.linear(i, w, b))

        output = torch.stack(output, dim=0)
        ensemble_output = ensemble_linear(input)

        assert output.equal(ensemble_output)

    @parameterized.expand([False, True])
    def test_ensemble_linear_correctness_with_broadcast_input(self, bias: bool) -> None:
        set_seed(SEED)

        tp_world_size = 8
        in_features = 7
        out_features = 11

        ensemble_linear = EnsembleLinear(in_features, out_features, tp_world_size, std=0.02, bias=bias)
        input = torch.randn(1, 4, 37, in_features)

        if bias:
            with torch.no_grad():
                nn.init.normal_(ensemble_linear.bias, mean=0, std=ensemble_linear.std)

        output = []
        for rank in range(tp_world_size):
            w = ensemble_linear.weight.view(tp_world_size, in_features, out_features)[rank, ...].T
            b = None if ensemble_linear.bias is None else ensemble_linear.bias[rank, :]

            output.append(F.linear(input.squeeze(0), w, b))

        output = torch.stack(output, dim=0)
        ensemble_output = ensemble_linear(input)

        assert output.equal(ensemble_output)
