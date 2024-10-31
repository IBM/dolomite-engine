import torch
import torch.nn.functional as F


def get_autoregressive_language_modeling_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    upcast_logits_for_loss: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_padding_free_transformer: bool = False,
) -> torch.Tensor:
    if use_padding_free_transformer:
        assert cu_seqlens is not None

        shift_logits = lm_logits[:-1, :]
        shift_labels = labels[1:].to(shift_logits.device)

        # this is needed so that the last token of current example doesn't predict first token of next example
        drop_loss_positions = cu_seqlens[1:-1] - 1
        shift_labels[drop_loss_positions] = -100
    else:
        assert cu_seqlens is None

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    if upcast_logits_for_loss:
        shift_logits = shift_logits.float()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss
