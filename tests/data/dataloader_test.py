from unittest import TestCase

from transformers import AutoConfig, AutoTokenizer

from dolomite_engine.arguments import TrainingArgs
from dolomite_engine.data import get_dataloader
from dolomite_engine.enums import DatasetSplit, Mode
from dolomite_engine.utils import load_yaml


class DataLoaderTest(TestCase):
    def test_dataloader(self) -> None:
        args = self.load_training_args_for_unit_tests()

        config = AutoConfig.from_pretrained(args.model_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)

        dataloader = get_dataloader(args, DatasetSplit.train, Mode.training, tokenizer, config.is_encoder_decoder)

        for example in dataloader:
            assert example["input_ids"].shape[0] == args.training_parameters.micro_batch_size
            break

    def load_training_args_for_unit_tests(self) -> TrainingArgs:
        return TrainingArgs(**load_yaml("tests/data/test_config.yml"))
