import os
from unittest import TestCase

from lm_engine.arguments import TrainingArgs
from lm_engine.utils import load_yaml


class TestCommons(TestCase):
    @staticmethod
    def load_training_args_for_unit_tests(filename: str) -> TrainingArgs:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        return TrainingArgs(**load_yaml(filepath))
