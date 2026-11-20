import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import DeepSeekMoE

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rotate_half(x):
    # GPT-NeoX style: пары (i, i + d/2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, dim // 2)
        # (T, dim) — дублируем freqs для пар (i, i + d/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, offset=0):
        # x: (B, T, dim) или (B, n_head, T, dim)
        dtype = x.dtype
        T = x.size(-2)
        cos = self.cos_cached[offset:offset + T]
        sin = self.sin_cached[offset:offset + T]
        if x.dim() == 3:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)              # (1, T, dim)
        else:
            cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
        x = x.float()
        return (x * cos + rotate_half(x) * sin).to(dtype)


class CausalAttention(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        default_head_dim = self.n_embed // self.n_head

        self.qk_nope_dim = getattr(config, "qk_nope_head_dim", default_head_dim)
        self.qk_rope_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_dim = getattr(config, "v_head_dim", self.qk_nope_dim)
        self.scale = (self.qk_nope_dim + self.qk_rope_dim) ** -0.5

        self.q_proj = nn.Linear(
            config.n_embed,
            self.n_head * (self.qk_nope_dim + self.qk_rope_dim),
            bias=False,
        )  # x_t @ W_Q -> [q_nope | q_rope] per head
        self.krope_proj = nn.Linear(config.n_embed, self.qk_rope_dim, bias=False)  # x_t @ W_KR (shared)
        self.rope = RotaryEmbedding(
            self.qk_rope_dim,
            base=getattr(config, "rope_theta", 10000.0),
            max_seq_len=config.block_size,
        )

        self.latent_dim = config.latent_dim
        self.wckv = nn.Linear(config.n_embed, self.latent_dim, bias=False)
        self.wuk = nn.Linear(self.latent_dim, self.n_head * self.qk_nope_dim, bias=False)
        self.wuv = nn.Linear(self.latent_dim, self.n_head * self.v_dim, bias=False)

        self.c_proj = nn.Linear(self.n_head * self.v_dim, config.n_embed, bias=False)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_size = config.block_size
        self.max_batch_size = config.max_batch_size

        # cache for latent kv + k_rope (их и храним вместо полных K/V)
        self.register_buffer(
            "latents",
            torch.zeros(self.max_batch_size, self.block_size, self.latent_dim, device=device),
            persistent=False,
        )
        self.register_buffer(
            "k_rope_cache",
            torch.zeros(self.max_batch_size, self.block_size, self.qk_rope_dim, device=device),
            persistent=False,
        )
        self.cache_len = 0

        # causal mask: 1 = can attend
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
            persistent=False,
        )

    def reset_cache(self):
        self.cache_len = 0

    def forward(self, x, use_cache=False):
        B, T, _ = x.size()
        offset = self.cache_len if use_cache else 0

        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.qk_nope_dim + self.qk_rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.krope_proj(x)
        k_rope = self.rope(k_rope, offset=offset)

        ckv = self.wckv(x)

        if use_cache:
            S = offset + T
            assert B <= self.max_batch_size and S <= self.block_size
            self.latents[:B, offset:S] = ckv.detach()
            self.k_rope_cache[:B, offset:S] = k_rope.detach()
            self.cache_len = S
            ckv = self.latents[:B, :S]
            k_rope = self.k_rope_cache[:B, :S]
        else:
            S = T

        uk_proj = self.wuk.weight.view(self.n_head, self.qk_nope_dim, self.latent_dim)
        Q_absorbed = torch.matmul(q_nope, uk_proj)

        attn_nope = torch.matmul(Q_absorbed, ckv.unsqueeze(-3).mT)
        attn_rope = torch.matmul(q_rope, k_rope.unsqueeze(1).mT)
        attn_weights = (attn_nope + attn_rope) * self.scale
        attn_weights = attn_weights.masked_fill(self.bias[:, :, offset:S, :S] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        v = self.wuv(ckv).view(B, S, self.n_head, self.v_dim).transpose(1, 2)
        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.v_dim)
        y = self.c_proj(y)

        return self.resid_dropout(y)


@dataclass
class Config:
    vocab_size: int = 256
    n_embed: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 128
    max_batch_size: int = 2

    # MLA
    latent_dim: int = 64          # kv_lora_rank
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    rope_theta: float = 10000.0

    # DeepSeekMoE
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    moe_intermediate_size: int = 128
    n_group: int = 2
    topk_group: int = 1
    topk_method: str = "group_limited_greedy"
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    aux_loss_alpha: float = 0.001
    dev_loss_alpha: float = 0.001
    seq_aux: bool = True
    mlp_bias: bool = False

    # dropout
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0


class DecoderLayer(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embed)
        self.attn = CausalAttention(config, device)
        self.ln_2 = RMSNorm(config.n_embed)
        self.moe = DeepSeekMoE(config)

    def forward(self, x, use_cache=False, return_aux=False):
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        h, aux = self.moe(self.ln_2(x), return_aux=True)
        x = x + h
        if return_aux:
            return x, aux
        return x


class DeepSeekV2(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        device = device if device is not None else torch.device("cpu")

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            DecoderLayer(config, device) for _ in range(config.n_layer)
        )
        self.ln_f = RMSNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("c_proj", "down_proj")):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        for layer in self.layers:
            layer.attn.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        self.reset_cache()
        logits, _ = self(idx[:, -self.config.block_size:], use_cache=True)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if temperature == 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if self.layers[0].attn.cache_len >= self.config.block_size:
                self.reset_cache()
                logits, _ = self(idx[:, -self.config.block_size:], use_cache=True)
            else:
                logits, _ = self(next_tok, use_cache=True)
        return idx

    def forward(self, idx, targets=None, use_cache=False, return_aux=False):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence longer than block_size"

        x = self.drop(self.tok_emb(idx))
        aux_sum = {"exp_bal": 0.0, "dev_bal": 0.0, "aux": 0.0}
        for layer in self.layers:
            x, aux = layer(x, use_cache=use_cache, return_aux=True)
            for k in aux_sum:
                aux_sum[k] = aux_sum[k] + aux[k]

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if return_aux:
                loss = loss + aux_sum["aux"]

        if return_aux:
            return logits, loss, aux_sum
        return logits, loss
