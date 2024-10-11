from ...utils import ProcessGroupManager
from ..utils import divide_if_divisible


def get_num_stages_on_current_rank(num_stages: int) -> tuple[int]:
    pp_world_size = ProcessGroupManager.get_pipeline_parallel_world_size()
    pp_rank = ProcessGroupManager.get_pipeline_parallel_rank()

    num_stages_per_rank = divide_if_divisible(
        num_stages, pp_world_size, f"num_stages {num_stages} must be evenly divisible by pp_world_size {pp_world_size}"
    )

    return tuple(pp_rank + i * pp_world_size for i in range(num_stages_per_rank))
