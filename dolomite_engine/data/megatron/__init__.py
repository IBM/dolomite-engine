import logging
from typing import List, Tuple

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ...arguments import TrainingArgs
from ...defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ...utils import get_global_rank, get_world_size, log_rank_0, print_rank_0
from .blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from .blended_megatron_dataset_config import GPTDatasetConfig
from .gpt_dataset import GPTDataset
from .utils import Split, compile_helpers


def get_megatron_gpt_dataloaders(args: TrainingArgs, tokenizer: AutoTokenizer, consumed_samples: int) -> None:
    assert len(args.datasets) == 1
    class_args = args.datasets[0].class_args

    assert args.datasets[0].max_input_tokens is None
    assert args.datasets[0].max_output_tokens is None
    assert args.datasets[0].max_output_tokens is None
    assert args.datasets[0].input_format == INPUT_FORMAT
    assert args.datasets[0].output_format == OUTPUT_FORMAT

    compile_helpers()

    log_rank_0(logging.INFO, "> building train, validation, and test datasets for GPT ...")
    print_rank_0()

    gpt_dataset_builder = BlendedMegatronDatasetBuilder(
        GPTDataset,
        sizes=_get_train_val_test_samples(
            args.training_parameters.num_training_steps,
            args.training_parameters.micro_batch_size,
            args.training_parameters.gradient_accumulation_steps,
            args.training_parameters.eval_interval,
            class_args.get("eval_steps"),
        ),
        config=GPTDatasetConfig(
            is_built_on_rank=_is_dataset_built_on_rank,
            random_seed=class_args.get("seed", args.random_args.seed),
            sequence_length=class_args.get("sequence_length"),
            blend=class_args.get("data_path"),
            blend_per_split=[
                class_args.get("train_data_path"),
                class_args.get("val_data_path"),
                class_args.get("test_data_path"),
            ],
            split=class_args.get("split"),
            path_to_cache=class_args.get("data_cache_path"),
            return_document_ids=False,
            fim_rate=class_args.get("fim_rate", 0),
            fim_spm_rate=class_args.get("fim_spm_rate", 0.5),
            node_uses_local_storage=class_args.get("node_uses_local_storage", False),
        ),
        tokenizer=tokenizer,
    )

    data_path = class_args.get("data_path")
    train_data_path = class_args.get("train_data_path")
    train_weighted_split_paths = class_args.get("train_weighted_split_paths")

    # Option 1: data loading using --data-path with single file
    # Option 2: data loading using --data-path with multiple weighted files
    # Option 3: data loading using --(train|val|test)-data-path with multiple weighted files
    if data_path is not None or train_data_path is not None:
        train_ds, val_ds, test_ds = gpt_dataset_builder.build()

        if not isinstance(val_ds, list):
            val_ds = [val_ds]
        if not isinstance(test_ds, list):
            test_ds = [test_ds]

    # Option 4: data loading using --(train|val|test)-weighted-split-paths
    elif train_weighted_split_paths:

        def _parse_and_get_dataset(weighted_split_paths: List[dict], dataset_split: Split) -> List[GPTDataset]:
            if weighted_split_paths is None:
                return []

            names = []
            paths = []
            splits = []
            weights = []
            for group in weighted_split_paths:
                assert len(group) == 1
                group_name = list(group.keys())[0]
                datasets_splits_list = group[group_name]

                names_ = []
                paths_ = []
                splits_ = []
                weights_ = []
                for d in datasets_splits_list:
                    names_.append(group_name)
                    paths_.append(d["path"])
                    splits_.append(d["split"])
                    weights_.append(d["weight"])

                names.append(names_)
                paths.append(paths_)
                splits.append(splits_)
                weights.append(weights_)

            return gpt_dataset_builder.build_dataset_single_split(names, paths, splits, weights, dataset_split)

        assert len(train_weighted_split_paths) == 1, "only 1 dataset group can be passed for training"
        train_ds = _parse_and_get_dataset(train_weighted_split_paths, Split.train)[0]

        val_ds = _parse_and_get_dataset(class_args.get("val_weighted_split_paths"), Split.valid)
        test_ds = _parse_and_get_dataset(class_args.get("test_weighted_split_paths"), Split.test)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0()
    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")
    print_rank_0()

    def _get_dataloader(dataset: GPTDataset, consumed_samples: int):
        if dataset is None:
            return None

        dataloader = DataLoader(
            dataset,
            batch_sampler=MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.training_parameters.micro_batch_size,
            ),
            num_workers=class_args.get("num_workers", 2),
            pin_memory=True,
        )

        return iter(dataloader)

    train_ds = _get_dataloader(train_ds, consumed_samples)
    val_ds = [_get_dataloader(i, 0) for i in val_ds]
    test_ds = [_get_dataloader(i, 0) for i in test_ds]

    return train_ds, val_ds, test_ds


def _is_dataset_built_on_rank() -> bool:
    return True


def _get_train_val_test_samples(
    num_training_steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    eval_interval: int,
    eval_steps: int,
) -> Tuple[int]:
    train_samples = num_training_steps * micro_batch_size * gradient_accumulation_steps * get_world_size()
    val_samples = (
        (num_training_steps // eval_interval + 1)
        * eval_steps
        * micro_batch_size
        * gradient_accumulation_steps
        * get_world_size()
    )
    test_samples = eval_steps * micro_batch_size * gradient_accumulation_steps * get_world_size()

    return train_samples, val_samples, test_samples


class MegatronPretrainingSampler:
    def __init__(
        self, total_samples: int, consumed_samples: int, micro_batch_size: int, drop_last: bool = True
    ) -> None:
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * get_world_size()
        self.drop_last = drop_last
        self.data_parallel_rank = get_global_rank()

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.consumed_samples < self.total_samples, "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.micro_batch_size > 0

    def __len__(self) -> int:
        return self.total_samples

    def _get_start_end_idx(self) -> Tuple[int, int]:
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self._get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self._get_start_end_idx()
            yield batch[start_idx:end_idx]
