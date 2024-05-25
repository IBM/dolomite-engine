from typing import Iterable, List

import torch

from ..enums import LossMask, Mode


def collate_fn(
    batch: List[dict],
    mode: Mode,
    loss_mask: LossMask,
    eos_token_id: int,
    is_encoder_decoder: bool,
    use_padding_free_transformer: bool,
    labels_mask_value: int = -100,
) -> dict:
    """prepares the batch with padding to pass into the forward function of the HuggingFace model

    Args:
        batch (Tuple[List[int]]): input tokens and output tokens. Output tokens are optional when running generation but required for training.

    Returns:
        dict: dict containing input_ids, attention_mask and labels if outputs is specified
    """

    inputs = [i["input"] for i in batch]
    outputs = [i["output"] for i in batch] if mode == Mode.training else None

    # labels is None when outputs is None
    labels = None

    if is_encoder_decoder:
        input_max_length = max(list(map(len, inputs)))

        input_ids = [[eos_token_id] * (input_max_length - len(array)) + array for array in inputs]
        attention_mask = [[0] * (input_max_length - len(array)) + [1] * len(array) for array in inputs]

        if outputs is not None:
            assert (
                loss_mask == LossMask.output_only
            ), "only output_only loss mask is supported with encoder decoder models"

            output_max_length = max(list(map(len, outputs)))
            # right padding for labels
            labels = [array + [labels_mask_value] * (output_max_length - len(array)) for array in outputs]
    else:
        if use_padding_free_transformer:
            input_ids = inputs
            attention_mask = None

            if loss_mask == LossMask.output_only:
                labels = [
                    [labels_mask_value] * (len(array_in) - len(array_out)) + array_out
                    for array_in, array_out in zip(inputs, outputs)
                ]
            elif loss_mask == LossMask.no_mask:
                labels = inputs
            else:
                raise ValueError(f"unexpected loss_mask ({loss_mask})")
        else:
            max_length = max(list(map(len, inputs)))

            input_ids = [[eos_token_id] * (max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in inputs]

            if outputs is not None:
                if loss_mask == LossMask.output_only:
                    labels = [[labels_mask_value] * (max_length - len(array)) + array for array in outputs]
                elif loss_mask == LossMask.no_mask:
                    labels = inputs
                else:
                    raise ValueError(f"unexpected loss_mask ({loss_mask})")

    if not use_padding_free_transformer:
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        if labels is not None:
            labels = torch.tensor(labels)

    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if mode == Mode.training:
        result["labels"] = labels

    return result


def infinite_iterator(x: Iterable) -> Iterable:
    """converts and iterable into a non-ending infinite iterable

    Args:
        x (Iterable): the iterable to convert

    Returns:
        Iterable: the converted iterable

    Yields:
        Iterator[Iterable]: an element from the original iterator
    """

    while True:
        for i in x:
            yield i
