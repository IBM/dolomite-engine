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
    @parameterized.expand(
        [
            ("dense_config.yml", "dense_mup.json", get_mup_group_with_names),
            ("moe_config.yml", "moe_mup.json", get_mup_group_with_names),
            ("dense_config.yml", "dense_normal.json", get_normal_group_with_names),
            ("moe_config.yml", "moe_normal.json", get_normal_group_with_names),
        ]
    )
    def test_mup_group(self, config_filename: str, expected_groups_filename: str, grouping_function: Callable) -> None:
        args = TestCommons.load_training_args_for_unit_tests(
            os.path.join("params_group/training_configs", config_filename)
        )

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
            ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
            ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ):
            model_container = get_model_container(args, Mode.training)

        _, names = grouping_function(model_container[0], args.optimizer_args.class_args)

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )
        assert expected_group == names
