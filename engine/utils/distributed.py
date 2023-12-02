import os
from typing import Callable

import torch.distributed as dist


def _get_int_env_var(var_name: str) -> int:
    """get the int value of an env variable, returns None if the variable is not found

    Args:
        var_name (str): env variable name

    Returns:
        int: value of the env variable
    """

    var_value = os.getenv(var_name)
    if var_value is not None:
        var_value = int(var_value)
    return var_value


def get_world_size() -> int:
    """number of GPUs

    Returns:
        int: number of GPUs
    """

    return _get_int_env_var("WORLD_SIZE")


def get_local_rank() -> int:
    """GPU rank on current node

    Returns:
        int: GPU rank on current node
    """

    return _get_int_env_var("LOCAL_RANK")


def get_rank() -> int:
    """GPU global rank across all nodes

    Returns:
        int: GPU global rank across all nodes
    """

    return _get_int_env_var("RANK")


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
            dist.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            dist.barrier()

    if get_rank() == rank:
        return func_rank_n
    elif get_rank() == None:
        # distributed is not initialized
        return func
    else:
        return func_rank_other
