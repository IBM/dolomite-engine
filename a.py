import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE_Grouped_Token(nn.Module):
    def __init__(
        self,
        dim,
        moe_sparsity=1 / 4,  # MoE sparsity
        expert_dim=256,
        n_total_experts=64,  # # activated expert = `topk_expert_fraction` * `n_total_experts`
        act=nn.SiLU(),
        need_sinkhorn=False,
        TC_routing=False,
        EC_routing=False,
        grouped_token_routing=True,
        n_grouped_token=64,
        group_max_deviation_from_EC=-1,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.act = act

        self.dim = dim
        self.expert_dim = expert_dim
        self.group_max_deviation_from_EC = group_max_deviation_from_EC

        print(f"Total width of this MoE: {n_total_experts * expert_dim}")

        assert moe_sparsity < 1 / 2, "assume the MoE sparsity is greater than 1/2"

        self.n_total_experts = n_total_experts
        self.n_topk_expert = int(moe_sparsity * n_total_experts)
        self.n_grouped_token = n_grouped_token

        self.need_sinkhorn = need_sinkhorn

        self.TC_routing = TC_routing
        self.EC_routing = EC_routing
        self.grouped_token_routing = grouped_token_routing

        # FP32 router, but should also work with BF16 router
        self.router = nn.Linear(dim, self.n_total_experts, bias=True, device=device)

        self.up = nn.ModuleList(
            [
                nn.Linear(dim, self.expert_dim, bias=True, device=device, dtype=dtype)
                for _ in range(self.n_total_experts)
            ]
        )
        self.gate = nn.ModuleList(
            [
                nn.Linear(dim, self.expert_dim, bias=True, device=device, dtype=dtype)
                for _ in range(self.n_total_experts)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Linear(self.expert_dim, dim, bias=True, device=device, dtype=dtype)
                for _ in range(self.n_total_experts)
            ]
        )

    def get_router_scores(self, x: torch.Tensor):
        router_logits = self.router(x.float())

        # router_scores for forward is `router_scores`
        router_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # router_scores_for_sorting does not need to be the same as `router_scores`
        router_scores_for_sorting = router_scores.clone()

        return router_scores, router_scores_for_sorting

    def sinkhorn_iteration(self, router_scores_for_sorting: torch.Tensor):
        for _ in range(10):
            # normalization over token  dimension
            router_scores_for_sorting /= router_scores_for_sorting.sum(dim=1, keepdim=True)

            # normalization over expert dimension
            router_scores_for_sorting /= router_scores_for_sorting.sum(dim=0, keepdim=True)
        return router_scores_for_sorting

    def get_group_top_indices(self, router_scores_for_sorting: torch.Tensor):
        T, E = router_scores_for_sorting.shape  # # token, # total expert
        K = self.n_topk_expert  # # activated expert
        G = self.n_grouped_token

        device = router_scores_for_sorting.device

        # first sorting, similar to EC
        router_sorting_sinkhorn_results = router_scores_for_sorting.sort(dim=0, descending=True)

        T_quotient = T // G
        T_remainder = T % G
        T_tilde = T_quotient * G

        if T_quotient == 0:
            E_summed_group_score = router_sorting_sinkhorn_results.values.sum(dim=0)
            selected_expert = E_summed_group_score.argmax()

            E_group_topk_indices = torch.stack(
                [
                    router_sorting_sinkhorn_results.indices[:, selected_expert],
                    torch.full(T, selected_expert, device=device),
                ],
                dim=-1,
            )

        # elif K == E # trivial case, all selected.

        else:
            if T_remainder == 0:
                grouping_selection_budget = T_quotient * K

                E_summed_group_score = router_sorting_sinkhorn_results.values.reshape(T_quotient, G, E).sum(dim=1)
            else:  # T is not fully divisble by quantization_step
                # ceiling the budget
                # we assume that K < 1/2 E, and quantization_selection_budget < T_quotient * E
                grouping_selection_budget = int(math.ceil(T * K / G))

                E_summed_group_score = (
                    router_sorting_sinkhorn_results.values[:T_tilde, :].reshape(T_quotient, G, E).sum(dim=1)
                )

            # second sorting
            if self.group_max_deviation_from_EC == -1:
                E_group_topk_mask = torch.zeros_like(E_summed_group_score, dtype=torch.bool)
                E_group_topk_mask.view(-1)[
                    E_summed_group_score.reshape(-1).topk(grouping_selection_budget).indices
                ] = True

            else:
                num_group_min = int(max((T_tilde * K / E) // G - self.group_max_deviation_from_EC, 0))
                num_group_max = int(min((T_tilde * K / E) // G + self.group_max_deviation_from_EC, T_tilde // G))

                grouping_selection_budget = grouping_selection_budget - num_group_min * E

                selectable_group_gap = num_group_max - num_group_min

                per_expert_selected_token_min = num_group_min * G
                per_expert_selected_token_max = num_group_max * G

                E_summed_group_score = (
                    router_sorting_sinkhorn_results.values[
                        per_expert_selected_token_min:per_expert_selected_token_max, :
                    ]
                    .reshape(selectable_group_gap, G, E)
                    .sum(dim=1)
                )

                # second sorting
                E_group_selectable_flatten_indices = (
                    E_summed_group_score.reshape(-1).topk(grouping_selection_budget).indices
                )

                E_group_topk_mask = torch.zeros(selectable_group_gap, E, dtype=torch.bool, device=device)

                # top selection that fits in the min-max margin, third sorting
                #   we want small indices here because the small indices correspond to largest value in the second sorting
                E_group_topk_mask.view(-1)[E_group_selectable_flatten_indices] = True

                E_group_topk_mask = torch.vstack(
                    [
                        torch.ones(num_group_min, E, dtype=torch.bool, device=device),
                        E_group_topk_mask,
                        torch.zeros(T_tilde // G - num_group_max, E, dtype=torch.bool, device=device),
                    ]
                )
                # E_group_topk_mask.sum(dim=0) will be n_grouped_token's multiples

            E_group_topk_mask = E_group_topk_mask[:, None, :].repeat(1, G, 1).reshape(T_tilde, E)

            if T_remainder == 0:
                E_group_topk_indices = torch.stack(
                    [
                        router_sorting_sinkhorn_results.indices[E_group_topk_mask],
                        torch.arange(0, E, device=device)[None, :].repeat(T, 1)[E_group_topk_mask],
                    ],
                    dim=-1,
                )
            else:
                E_group_topk_indices = torch.stack(
                    [
                        router_sorting_sinkhorn_results.indices[:T_tilde][E_group_topk_mask],
                        torch.arange(0, E, device=device)[None, :].repeat(T_tilde, 1)[E_group_topk_mask],
                    ],
                    dim=-1,
                )

        # E_group_topk_indices: T, E. The selected tokens for each expert is a multiple of `n_grouped_token`
        return E_group_topk_indices

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        B, L, dim = x.shape
        T = B * L
        D = dim

        K = self.n_topk_expert
        E = self.n_total_experts

        x = x.reshape(T, dim)

        router_scores, router_scores_for_sorting = self.get_router_scores(x)
        if self.need_sinkhorn:
            router_scores_for_sorting = self.sinkhorn_iteration(router_scores_for_sorting)

        if self.TC_routing:
            routed_experts_topk = router_scores_for_sorting.topk(K, dim=1)
            indices = routed_experts_topk.indices
            scores = routed_experts_topk.values
            scores /= scores.sum(keepdim=True, dim=1)

            forward_scores = torch.zeros_like(router_scores)
            forward_scores.scatter_(1, indices, scores)

        elif self.EC_routing:
            # some EC impl will take softmax over token dimension
            #   here I still take softmax over expert dimension and only sort over the token dimension

            num_token_expert_choice = T * K // E

            token_ranking = router_scores_for_sorting.topk(num_token_expert_choice, dim=0)

            expert_choice_indices = torch.stack(
                [
                    token_ranking.indices.view(-1),
                    torch.arange(self.n_total_experts, device=x.device)[None, :]
                    .repeat(num_token_expert_choice, 1)
                    .view(-1),
                ],
                dim=-1,
            )

            forward_scores = torch.zeros_like(router_scores)
            forward_scores[expert_choice_indices[..., 0], expert_choice_indices[..., 1]] = router_scores[
                expert_choice_indices[..., 0], expert_choice_indices[..., 1]
            ]

            # final normalization over expert dimension
            forward_scores = forward_scores / (forward_scores.sum(dim=1, keepdim=True) + 1e-6)

        elif self.grouped_token_routing:
            E_grouped_top_indices = self.get_group_top_indices(router_scores_for_sorting)

            forward_scores = torch.zeros_like(router_scores)

            forward_scores[E_grouped_top_indices[..., 0], E_grouped_top_indices[..., 1]] = router_scores[
                E_grouped_top_indices[..., 0], E_grouped_top_indices[..., 1]
            ]

            # eps = 1e-6 to avoid division by 0
            forward_scores = forward_scores / (forward_scores.sum(keepdim=True, dim=-1) + 1e-6)

        else:
            raise NotImplementedError()

        output = torch.zeros_like(x)
        for expert_idx in range(self.n_total_experts):
            up_expert = self.up[expert_idx]
            gate_expert = self.gate[expert_idx]
            down_expert = self.down[expert_idx]

            selected_token = torch.argwhere(forward_scores[:, expert_idx] > 0).flatten()
            selected_x = x[selected_token, :]
            score = forward_scores[selected_token, expert_idx].to(input_dtype)

            expert_output = down_expert(self.act(up_expert(selected_x)) * gate_expert(selected_x))

            output[selected_token] += expert_output * score.unsqueeze(-1)

        return output.reshape(B, L, D)

    def extra_repr(self):
        if self.TC_routing:
            routing = "token choice"
        elif self.EC_routing:
            routing = "expert choice"
        elif self.grouped_token_routing:
            routing = f"grouped-token routing with size ({self.n_grouped_token})"

        return f"{routing}: expert_dim={self.expert_dim}, n_total_experts={self.n_total_experts}, n_topk_experts={self.n_topk_expert}"


if __name__ == "__main__":
    B = 256  # batch size
    L = 1024  # seq length
    D = 768  # hidden dim
    E = 128  # num of experts
    D_E = 256  # expert width

    print(f"total width {D_E * E}, equivalent as {int((D_E * E) / (D * 8/3))} of standard MoE")
    n_grouped_token = 64
    moe_sparsity = 1 / 8

    MoE = MoE_Grouped_Token(
        dim=D,
        moe_sparsity=moe_sparsity,
        expert_dim=D_E,
        n_total_experts=E,
        need_sinkhorn=False,
        TC_routing=False,
        EC_routing=False,
        grouped_token_routing=True,
        n_grouped_token=n_grouped_token,
        group_max_deviation_from_EC=-1,
    )

    x = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
    y = MoE(x)
