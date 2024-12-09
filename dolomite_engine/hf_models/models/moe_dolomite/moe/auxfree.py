import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._functional_collectives import all_reduce

from .....utils import ProcessGroupManager
from ..config import MoEDolomiteConfig
from .scatter import ScatterMoE


class AuxFreeMoE(ScatterMoE):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        super().__init__(config, use_padding_free_transformer, layer_idx)
        self.register_buffer("bias", torch.zeros(config.num_experts))
        self.step_size = config.router_aux_loss_coef

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        router_logits = self.gate(hidden_states)

        with torch.no_grad():
            _, selected_experts = self._get_topk(F.softmax(router_logits, dim=-1) + self.bias)

        router_weights = router_logits[
            torch.arange(hidden_states.size(0), device=hidden_states.device, dtype=torch.int32)[:, None],
            selected_experts,
        ]
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_switch_loss(self, logits: Tensor, probs: Tensor, topk_idxs: Tensor) -> Tensor:
        num_experts = logits.size(-1)
        freq = torch.bincount(topk_idxs.flatten(), minlength=num_experts).to(dtype=logits.dtype)

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            freq = all_reduce(freq, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group())

        avg_counts = torch.mean(freq, dim=0, keepdim=True)

        if self.training and self.step_size > 0:
            self.bias += self.step_size * torch.sign(avg_counts - freq)

        with torch.no_grad():
            acc_probs = probs.sum(0)
            switch_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()

        return switch_loss.detach()
