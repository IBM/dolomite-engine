import os
from typing import Callable

import torch.distributed


_WORLD_SIZE: int = None
_LOCAL_RANK: int = None
_GLOBAL_RANK: int = None


def get_world_size() -> int:
    """number of GPUs

    Returns:
        int: number of GPUs
    """

    global _WORLD_SIZE

    if _WORLD_SIZE is None:
        _WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    return _WORLD_SIZE


def get_local_rank() -> int:
    """GPU rank on current node

    Returns:
        int: GPU rank on current node
    """

    global _LOCAL_RANK

    if _LOCAL_RANK is None:
        _LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    return _LOCAL_RANK


def get_global_rank() -> int:
    """GPU global rank across all nodes

    Returns:
        int: GPU global rank across all nodes
    """

    global _GLOBAL_RANK

    if _GLOBAL_RANK is None:
        _GLOBAL_RANK = int(os.getenv("RANK", 0))
    return _GLOBAL_RANK


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> Callable:
    """wraps a function to run on a single rank, returns a no-op for other ranks

    Args:
        func (Callable): function to wrap
        rank (int, optional): rank on which function should run. Defaults to 0.
        barrier (bool, optional): whether to synchronize the processes at the end of function execution. Defaults to False.

    Returns:
        Callable: wrapped function
    """

    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            torch.distributed.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            torch.distributed.barrier()

    if get_global_rank() == rank:
        return func_rank_n
    elif get_global_rank() == None:
        # distributed is not initialized
        return func
    else:
        return func_rank_other
