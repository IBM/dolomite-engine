import json
import os

import torch
from parameterized import parameterized

from dolomite_engine.enums import Mode, ParamsGroupMethod
from dolomite_engine.model_wrapper import get_model_container
from dolomite_engine.optimization.params_group import get_param_groups_list
from dolomite_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class ParamsGroupTest(TestCommons):
    @parameterized.expand([("mup.json", ParamsGroupMethod.mup), ("normal.json", None)])
    def test_mup_group(self, expected_groups_filename: str, params_group_method: ParamsGroupMethod) -> None:
        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
            ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
            ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ):
            model_container = get_model_container(args, Mode.training)

        params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, params_group_method)[0]

        # json.dump(params_groups.get_param_names(), open("a.json", "w"), indent=4)

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )
        assert expected_group == params_groups.get_param_names()
