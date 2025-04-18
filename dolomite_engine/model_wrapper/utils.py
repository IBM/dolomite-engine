import torch

from ..utils import ProcessGroupManager


def broadcast_tensor_parallel_input(tokens: dict, shape: tuple[int]) -> torch.Tensor:
    if ProcessGroupManager.is_tensor_parallel_first_rank():
        tokens = tokens.to(torch.cuda.current_device())
    else:
        tokens = torch.empty(shape, dtype=torch.long, device=torch.cuda.current_device())

    torch.distributed.broadcast(
        tokens,
        src=ProcessGroupManager.get_tensor_parallel_first_rank(),
        group=ProcessGroupManager.get_tensor_parallel_group(),
    )

    return tokens


def split_main_and_mtp_inputs(
    input_ids: torch.Tensor,
    num_mtp_modules: int,
):
    """
    Split input tensors into the main TRM blocks and MTP block forward pass

    Here the Full Seq_len = Actual Seq_len (T) + num_mtp_modules + 1
    """

    # Get the T based on input_ids/input_embeds

    full_seq_len = input_ids.shape[1]
    T = full_seq_len - (num_mtp_modules + 1)

    def get_tensor_slice(tensor, start, end):
        return None if tensor is None else tensor[:, start:end]

    main_trm = {
        "input_ids": get_tensor_slice(input_ids, 0, T),
        "labels": get_tensor_slice(input_ids, 1, T + 1),
    }

    mtp_inp_list = []

    for i in range(1, num_mtp_modules + 1):
        mtp_ip = {
            "input_ids": get_tensor_slice(input_ids, i, T + i),
            "labels": get_tensor_slice(input_ids, i + 1, T + i + 1),
        }

        mtp_inp_list.append(mtp_ip)

    return main_trm, mtp_inp_list
