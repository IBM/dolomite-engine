import torch


def divide_if_divisible(dividend: int, divisor: int, msg: str) -> int:
    """divide if divisible else raise an error

    Args:
        dividend (int): dividend
        divisor (int): divisor
        msg (str): error message

    Returns:
        int: result
    """

    assert dividend % divisor == 0, msg
    return dividend // divisor


def convert_padding_free_lists_to_tensors(
    input_ids: list[list[int]] | None,
    inputs_embeds: list[list[float]] | None,
    position_ids: list[list[int]] | None,
    token_type_ids: list[list[int]] | None,
    labels: list[list[int]] | None,
) -> tuple[torch.Tensor]:
    device = torch.cuda.current_device()

    # check input types are correct
    error_message = "{variable} should be of type List[List[{dtype}]]"
    _check_list_type(input_ids, error_message.format(variable="input_ids", dtype="int"))
    _check_list_type(inputs_embeds, error_message.format(variable="inputs_embeds", dtype="float"))
    _check_list_type(position_ids, error_message.format(variable="position_ids", dtype="int"))
    _check_list_type(token_type_ids, error_message.format(variable="token_type_ids", dtype="int"))
    _check_list_type(labels, error_message.format(variable="labels", dtype="int"))

    # prepare inputs for the model
    seqlens = torch.tensor([0] + [len(x) for x in input_ids])
    cu_seqlens = seqlens.cumsum(dim=-1).to(device, torch.int32)
    max_seqlen = seqlens.max().to(device)

    if position_ids is None:
        position_ids = [list(range(len(x))) for x in input_ids]
    position_ids = _flatten_and_convert_to_tensors(position_ids, device)

    input_ids = _flatten_and_convert_to_tensors(input_ids, device)

    if inputs_embeds is not None:
        inputs_embeds = _flatten_and_convert_to_tensors(inputs_embeds, device)

    if token_type_ids is not None:
        token_type_ids = _flatten_and_convert_to_tensors(token_type_ids, device)

    if labels is not None:
        labels = _flatten_and_convert_to_tensors(labels, device)

    return input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen


def _check_list_type(list_of_list: list[list[int | float]] | None, error_message: str) -> None:
    if list_of_list is None:
        return

    assert isinstance(list_of_list, list), error_message
    assert isinstance(list_of_list[0], list), error_message


def _flatten_and_convert_to_tensors(x: list[int], device: torch.device) -> torch.Tensor:
    y = []
    for sequence in x:
        y.extend(sequence)

    return torch.tensor(y, device=device)
