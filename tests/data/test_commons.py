from unittest import TestCase

from dolomite_engine.arguments import TrainingArgs
from dolomite_engine.utils import load_yaml


class TestCommons(TestCase):
    @staticmethod
    def load_training_args_for_unit_tests() -> TrainingArgs:
        return TrainingArgs(**load_yaml("tests/data/test_config.yml"))
