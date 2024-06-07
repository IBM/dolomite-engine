# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from enum import Enum
from typing import List

import numpy
import torch
from torch.utils.cpp_extension import load as load_cpp_extension

from ....utils import log_rank_0


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers() -> None:
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""

    log_rank_0(logging.INFO, "compiling helpers.cpp")

    if torch.cuda.current_device() == 0:
        load_cpp_extension(
            "helpers",
            sources=os.path.join(os.path.dirname(__file__), "helpers.cpp"),
            extra_cflags=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC", "-fdiagnostics-color"],
        )

    torch.distributed.barrier()


def build_blending_indices(
    dataset_index: numpy.ndarray,
    dataset_sample_index: numpy.ndarray,
    weights: List[float],
    num_datasets: int,
    size: int,
    verbose: bool,
) -> None:
    import helpers

    helpers.build_blending_indices(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose)


def build_sample_idx(
    sizes: numpy.ndarray, doc_idx: numpy.ndarray, sequence_length: int, num_epochs: int, tokens_per_epoch: int
) -> numpy.ndarray:
    import helpers

    return helpers.build_sample_idx(sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w
