import torch
import torch.nn as nn

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
