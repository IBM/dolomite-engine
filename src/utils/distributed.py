import os
from typing import Callable

import torch.distributed as dist


def _get_int_env_var(var_name: str) -> int:
    var_value = os.getenv(var_name)
    if var_value is not None:
        var_value = int(var_value)
    return var_value


def get_world_size() -> int:
    return _get_int_env_var("WORLD_SIZE")


def get_local_rank() -> int:
    return _get_int_env_var("LOCAL_RANK")


def get_rank() -> int:
    return _get_int_env_var("RANK")


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> None:
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
