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
        # B - batch size, T - seq length, c - dimension of the vector

        q = self.q_proj(x)  # x_t @ W_Q
        ckv = self.wckv(x)  # C_KV = x_t @ W_DKV

        uk_proj = 
