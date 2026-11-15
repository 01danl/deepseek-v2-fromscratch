import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.q_proj = nn.Linear(config.n_embd, config.n_embd) # Linear(in, out)

        self.latent_dim = config.latent_dim
        self.wckv = nn.Linear(config.n_embd, self.latent_dim)
        self.wuk = nn.Linear(self.latent_dim, config.n_embed)
        self.wuv = nn.Linear(self.latent_dim, config.n_embed)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_size = config.block_size
        self.max_batch_size = config.max_batch_size

        # cache for latent kv
        self.latents = torch.zeros(
            (self.max_batch_size, self.block_size, self.latent_dim), device=device
        )
        self.cache_len = 0

        assert self.n_embed % self.n_head == 0
        self.head_dim = self.n_embed // self.n_head

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

        q = self.q_proj(x)  # x_t @ W_Q
        c_kv = self.wckv(x)  # C_KV = x_t @ W_DKV

        if use_cache:
            assert self.cache_len + T <= self.block_size, "KV cache overflow"
            past = self.cache_len
            self.latents[:B, past : past + T] = c_kv.detach()
            c_all = self.latents[:B, : past + T]  # (B, S, latent_dim)
            self.cache_len += T
        else:
            past = 0
            c_all = c_kv

        S = c_all.size(1)  # total key length = past + T

        # up-project latents to full K and V
        k = self.wuk(c_all)  # (B, S, C)
        v = self.wuv(c_all)  # (B, S, C)

        # split into heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hd)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, S, hd)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, S)

        # queries sit at positions past..past+T-1, keys at 0..S-1
        mask = self.bias[:, :, past : past + T, :S]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))
