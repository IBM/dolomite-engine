import json
import os
from typing import Callable

import torch
from parameterized import parameterized

from dolomite_engine.enums import Mode
from dolomite_engine.model_wrapper import get_model_container
from dolomite_engine.optimization.params_group import get_mup_group_with_names, get_normal_group_with_names
from dolomite_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class ParamsGroupTest(TestCommons):
    @parameterized.expand([("mup.json", get_mup_group_with_names), ("normal.json", get_normal_group_with_names)])
    def test_mup_group(self, expected_groups_filename: str, grouping_function: Callable) -> None:
        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
            ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
            ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ):
            model_container = get_model_container(args, Mode.training)

        names = grouping_function(model_container[0], args.optimizer_args.class_args)
        resulting_groups = {}
        for group_name in names:
            resulting_groups[group_name] = []
            for param_name in names[group_name]["parameter_map"]:
                resulting_groups[group_name].append(param_name)
            resulting_groups[group_name].sort()

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )
        assert expected_group == resulting_groups
