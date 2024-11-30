import json
import os

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
            ("gpt_dolomite_config.yml", "gpt_dolomite_mup.json"),
            ("moe_dolomite_config.yml", "moe_dolomite_mup.json"),
            ("rnn_dolomite_config.yml", "rnn_dolomite_mup.json"),
        ]
    )
    def test_mup_group(self, config_filename: str, expected_groups_filename: str) -> None:
        args = TestCommons.load_training_args_for_unit_tests(
            os.path.join("params_group/training_configs", config_filename)
        )

        if "rnn_dolomite" in config_filename:
            if not torch.cuda.is_available():
                self.skipTest("skipping test because CUDA is unavailable")

            try:
                with (
                    torch.device("meta"),
                    ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
                    ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
                    ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
                    ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
                ):
                    model_container = get_model_container(args, Mode.training)
            except RuntimeError:
                self.skipTest("skipping rnn_dolomite test since causal-conv1d is not installed")
        else:
            with (
                torch.device("meta"),
                ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
                ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
                ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
                ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
            ):
                model_container = get_model_container(args, Mode.training)

        _, names = get_mup_group_with_names(model_container[0], args.optimizer_args.class_args)

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )
        assert expected_group == names

    @parameterized.expand(
        [
            ("gpt_dolomite_config.yml", "gpt_dolomite_normal.json"),
            ("moe_dolomite_config.yml", "moe_dolomite_normal.json"),
            ("rnn_dolomite_config.yml", "rnn_dolomite_normal.json"),
        ]
    )
    def test_normal_group(self, config_filename: str, expected_groups_filename: str) -> None:
        args = TestCommons.load_training_args_for_unit_tests(
            os.path.join("params_group/training_configs", config_filename)
        )

        if "rnn_dolomite" in config_filename:
            if not torch.cuda.is_available():
                self.skipTest("skipping test because CUDA is unavailable")

            try:
                with (
                    torch.device("meta"),
                    ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
                    ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
                    ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
                    ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
                ):
                    model_container = get_model_container(args, Mode.training)
            except RuntimeError:
                self.skipTest("skipping rnn_dolomite test since causal-conv1d is not installed")
        else:
            with (
                torch.device("meta"),
                ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
                ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
                ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
                ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
            ):
                model_container = get_model_container(args, Mode.training)

        _, names = get_normal_group_with_names(model_container[0], args.optimizer_args.class_args)

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )
        assert expected_group == names
