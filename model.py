import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    RoPE для decoupled rotary части MLA (q_rope / k_rope).
    Применяется ТОЛЬКО к rope-части головы, не к q_nope/k_nope:
    RoPE зависит от позиции и поэтому не коммутирует с absorption.
    """
    def __init__(self, dim, base=10000.0, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, dim // 2)
        # (T, dim) — дублируем freqs для пар (i, i + d/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def apply(self, x, offset=0):
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

        assert self.n_embed % self.n_head == 0
        self.head_dim = self.n_embed // self.n_head

        # decoupled RoPE (как в DeepSeek V2): голова = head_dim (nope, без позиции,
        # участвует в absorption) + qk_rope_dim (rope, с RoPE). k_rope — одна
        # проекция, общая для всех голов (MQA).
        self.qk_rope_dim = getattr(config, "qk_rope_head_dim", 64)
        self.scale = (self.head_dim + self.qk_rope_dim) ** -0.5

        self.q_proj = nn.Linear(
            config.n_embed,
            config.n_head * (self.head_dim + self.qk_rope_dim),
            bias=False,
        )  # x_t @ W_Q -> [q_nope | q_rope] per head
        self.krope_proj = nn.Linear(config.n_embed, self.qk_rope_dim, bias=False)  # x_t @ W_KR (shared)
        self.rope = RotaryEmbedding(
            self.qk_rope_dim,
            base=getattr(config, "rope_theta", 10000.0),
            max_seq_len=config.block_size,
        )

        self.latent_dim = config.latent_dim
        self.wckv = nn.Linear(config.n_embed, self.latent_dim)
        self.wuk = nn.Linear(self.latent_dim, config.n_embed)
        self.wuv = nn.Linear(self.latent_dim, config.n_embed)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_size = config.block_size
        self.max_batch_size = config.max_batch_size

        # cache for latent kv + k_rope (их и храним вместо полных K/V)
        self.latents = torch.zeros(
            (self.max_batch_size, self.block_size, self.latent_dim), device=device
        )
        self.k_rope_cache = torch.zeros(
            (self.max_batch_size, self.block_size, self.qk_rope_dim), device=device
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
        B, T, C = x.size()
        # B - batch size, T - seq length, c - dimension of the vector

        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim + self.qk_rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.head_dim, self.qk_rope_dim], dim=-1)

        # RoPE только на rope-части (decoupled), q_nope не трогаем —
        # он участвует в absorption и должен оставаться position-free
        q_rope = self.rope.apply(q_rope)  # (B, h, T, rope)
        k_rope = self.krope_proj(x)       # (B, T, rope) — общий для всех голов
        k_rope = self.rope.apply(k_rope)  # (B, T, rope)

        ckv = self.wckv(x)  # C_KV = x_t @ W_DKV

        #every head gets its own projection slice
        uk_proj = self.wuk.weight.view(self.n_head, self.head_dim, self.latent_dim)
        uv_proj = self.wuv.weight.view(self.n_head, self.head_dim, self.latent_dim)

        Q_absorbed = torch.matmul(q_nope, uk_proj)  # (B, h, T, latent_dim)

        # score = q_nope * k_nope^T (absorption) + q_rope * k_rope^T (RoPE)
        attn_nope = torch.matmul(Q_absorbed, ckv.unsqueeze(-3).mT)  # (B, h, T, S)
        attn_rope = torch.matmul(q_rope, k_rope.unsqueeze(1).mT)    # (B, h, T, S), k_rope broadcast по головам
        attn_weights = (attn_nope + attn_rope) * self.scale
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        return attn_weights 
