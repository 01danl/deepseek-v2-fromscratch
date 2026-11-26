import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim, intermediate, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=bias)
        self.up_proj = nn.Linear(dim, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, dim, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """
    V2: softmax по всем экспертам, device-limited routing (max по группе),
    затем top-K внутри выбранных групп. Веса — сырые вероятности softmax,
    умноженные на routed_scaling_factor (norm_topk_prob = false).
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.scaling = getattr(config, "routed_scaling_factor", 1.0)
        self.norm_topk = getattr(config, "norm_topk_prob", False)
        self.topk_method = getattr(config, "topk_method", "greedy")
        self.group_size = self.n_experts // self.n_group

        assert self.n_experts % self.n_group == 0, "n_routed_experts must divide n_group"

        self.weight = nn.Parameter(torch.empty(self.n_experts, config.n_embed))
        nn.init.normal_(self.weight, std=0.006)

    def forward(self, x):
        probs = F.linear(x.float(), self.weight.float()).softmax(dim=-1)

        scores = probs
        if self.topk_method == "group_limited_greedy":
            group_scores = probs.view(-1, self.n_group, self.group_size).max(dim=-1).values
            group_idx = group_scores.topk(self.topk_group, dim=-1).indices
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
            mask = group_mask.unsqueeze(-1).expand(-1, -1, self.group_size).reshape(-1, self.n_experts)
            scores = probs.masked_fill(mask == 0, float("-inf"))

        weight, idx = scores.topk(self.top_k, dim=-1)
        if self.norm_topk:
            weight = weight / weight.sum(dim=-1, keepdim=True)
        return idx, weight * self.scaling, probs


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE из V2: N_s shared экспертов (всегда активны) + N_r routed,
    из которых выбираются top-K. Fine-grained сегментация: эксперты мелкие
    (moe_intermediate_size), поэтому их много.
    Shared реализован одним FFN с intermediate * n_shared_experts —
    это эквивалентно сумме независимых shared-экспертов.
    """

    def __init__(self, config):
        super().__init__()
        d = config.n_embed
        m = getattr(config, "moe_intermediate_size", 4 * d)
        bias = getattr(config, "mlp_bias", False)

        self.n_routed = config.n_routed_experts
        self.n_shared = getattr(config, "n_shared_experts", 2)
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.alpha1 = getattr(config, "aux_loss_alpha", 0.001)
        self.alpha2 = getattr(config, "dev_loss_alpha", 0.001)
        self.alpha3 = getattr(config, "comm_loss_alpha", 0.001)
        self.seq_aux = getattr(config, "seq_aux", True)

        self.experts = nn.ModuleList(Expert(d, m, bias=bias) for _ in range(self.n_routed))
        self.gate = MoEGate(config)
        self.shared_experts = (
            Expert(d, m * self.n_shared, bias=bias) if self.n_shared > 0 else None
        )

    def _aux_losses(self, probs, idx, B, T):
        E = self.n_routed
        K = self.top_k
        G = self.n_group
        gs = self.gate.group_size

        if self.seq_aux:
            counts = F.one_hot(idx.view(B, T * K), E).float().sum(dim=-2)
            f = counts * E / (T * K)
            P = probs.view(B, T, E).mean(dim=1)
            exp_bal = self.alpha1 * (f * P).sum(dim=-1).mean()
            f_g = f.view(B, G, -1).mean(dim=-1)
            P_g = P.view(B, G, -1).sum(dim=-1)
            dev_bal = self.alpha2 * (f_g * P_g).sum(dim=-1).mean()

            sent = F.one_hot(idx.view(B, T, K) // gs, G).float().amax(dim=-2)
            f_sent = sent.sum(dim=1) * G / (T * self.topk_group)
            comm_bal = self.alpha3 * (f_sent * P_g).sum(dim=-1).mean()
        else:
            counts = F.one_hot(idx.reshape(-1), E).float().mean(dim=0)
            f = counts * E
            P = probs.mean(dim=0)
            exp_bal = self.alpha1 * (f * P).sum()
            f_g = f.view(G, -1).mean(dim=-1)
            P_g = P.view(G, -1).sum(dim=-1)
            dev_bal = self.alpha2 * (f_g * P_g).sum()

            sent = F.one_hot(idx.reshape(-1, K) // gs, G).float().amax(dim=1)
            f_sent = sent.mean(dim=0) * G / self.topk_group
            comm_bal = self.alpha3 * (f_sent * P_g).sum()

        return {
            "exp_bal": exp_bal,
            "dev_bal": dev_bal,
            "comm_bal": comm_bal,
            "aux": exp_bal + dev_bal + comm_bal,
        }

    def forward(self, x, return_aux=False):
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)

        idx, weight, probs = self.gate(x_flat)
        weight = weight.to(x.dtype)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            tok, slot = (idx == i).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            y.index_add_(0, tok, expert(x_flat[tok]) * weight[tok, slot].unsqueeze(-1))

        if self.shared_experts is not None:
            y = y + self.shared_experts(x_flat)

        y = y.view(B, T, d)
        if return_aux:
            return y, self._aux_losses(probs, idx, B, T)
        return y
