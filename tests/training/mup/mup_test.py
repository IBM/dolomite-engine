import json
import os

import torch

from dolomite_engine.enums import Mode, ParamsGroupMethod
from dolomite_engine.model_wrapper import get_model
from dolomite_engine.optimization.optimizer import _get_param_groups
from dolomite_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class MuPTest(TestCommons):
    def test_gpt_dolomite_mup(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests("mup/gpt_dolomite.yml")

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ):
            model = get_model(args, Mode.training)

        _, names = _get_param_groups(model, args.optimizer_args.class_args, params_group_method=ParamsGroupMethod.mup)

        expected_group = json.load(open(os.path.join(os.path.dirname(__file__), "gpt_dolomite.json"), "r"))
        assert expected_group == names

    def test_moe_dolomite_mup(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests("mup/moe_dolomite.yml")

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ):
            model = get_model(args, Mode.training)

        _, names = _get_param_groups(model, args.optimizer_args.class_args, params_group_method=ParamsGroupMethod.mup)

        expected_group = json.load(open(os.path.join(os.path.dirname(__file__), "moe_dolomite.json"), "r"))
        assert expected_group == names
