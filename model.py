from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import DeepSeekMoE, Expert


@dataclass
class Config:
    # attention / MLA
    n_embed: int = 5120
    n_head: int = 128
    latent_dim: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rope_scaling_mscale: float = 1.0
    block_size: int = 4096
    max_batch_size: int = 1
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0

    # transformer
    num_hidden_layers: int = 60
    first_k_dense_replace: int = 1
    intermediate_size: int = 512
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.006
    vocab_size: int = 102400
    tie_word_embeddings: bool = False

    # MoE
    n_routed_experts: int = 160
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 1536
    n_group: int = 8
    topk_group: int = 3
    routed_scaling_factor: float = 16.0
    norm_topk_prob: bool = False
    topk_method: str = "group_limited_greedy"
    aux_loss_alpha: float = 0.001
    dev_loss_alpha: float = 0.001
    comm_loss_alpha: float = 0.001
    seq_aux: bool = True


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
        default_head_dim = self.n_embed // self.n_head
        eps = getattr(config, "rms_norm_eps", 1e-6)

        # decoupled RoPE (как в DeepSeek V2): голова = qk_nope (без позиции,
        # участвует в absorption) + qk_rope (с RoPE). k_rope — одна
        # проекция, общая для всех голов (MQA).
        self.qk_nope_dim = getattr(config, "qk_nope_head_dim", default_head_dim)
        self.qk_rope_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_dim = getattr(config, "v_head_dim", self.qk_nope_dim)
        self.scale = ((self.qk_nope_dim + self.qk_rope_dim) ** -0.5) * (
            getattr(config, "rope_scaling_mscale", 1.0) ** 2
        )

        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        if self.q_lora_rank is not None:
            self.wq_a = nn.Linear(config.n_embed, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank,
                self.n_head * (self.qk_nope_dim + self.qk_rope_dim),
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                config.n_embed,
                self.n_head * (self.qk_nope_dim + self.qk_rope_dim),
                bias=False,
            )

        self.krope_proj = nn.Linear(config.n_embed, self.qk_rope_dim, bias=False)
        self.rope = RotaryEmbedding(
            self.qk_rope_dim,
            base=getattr(config, "rope_theta", 10000.0),
            max_seq_len=config.block_size,
        )

        self.latent_dim = config.latent_dim
        self.wckv = nn.Linear(config.n_embed, self.latent_dim, bias=False)
        self.kv_norm = RMSNorm(self.latent_dim, eps=eps)
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
        )
        self.register_buffer(
            "k_rope_cache",
            torch.zeros(self.max_batch_size, self.block_size, self.qk_rope_dim, device=device),
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

    def _ensure_cache(self, B, S, dtype, device):
        b, s = self.latents.shape[0], self.latents.shape[1]
        if B > b or S > s or self.latents.dtype != dtype or self.latents.device != device:
            b, s = max(B, b), max(S, s)
            self.latents = torch.zeros(b, s, self.latent_dim, dtype=dtype, device=device)
            self.k_rope_cache = torch.zeros(b, s, self.qk_rope_dim, dtype=dtype, device=device)

    def forward(self, x, use_cache=False):
        B, T, _ = x.size()
        dtype, device = x.dtype, x.device
        offset = self.cache_len if use_cache else 0
        S = self.cache_len + T if use_cache else T

        if self.q_lora_rank is not None:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.qk_nope_dim + self.qk_rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # RoPE только на rope-части (decoupled), q_nope не трогаем —
        # он участвует в absorption и должен оставаться position-free
        q_rope = self.rope.apply(q_rope, offset=offset)  # (B, h, T, rope)
        k_rope_new = self.rope.apply(self.krope_proj(x), offset=offset)  # (B, T, rope)

        # C_KV = norm(x_t @ W_DKV), нормализация — до записи в кэш и до attn/v
        ckv_new = self.kv_norm(self.wckv(x))

        if use_cache:
            self._ensure_cache(B, S, dtype, device)
            self.latents[:B, offset:S] = ckv_new.detach()
            self.k_rope_cache[:B, offset:S] = k_rope_new.detach()
            ckv = self.latents[:B, :S]
            k_rope = self.k_rope_cache[:B, :S]
        else:
            ckv = ckv_new
            k_rope = k_rope_new

        #every head gets its own projection slice
        uk_proj = self.wuk.weight.view(self.n_head, self.qk_nope_dim, self.latent_dim)

        Q_absorbed = torch.matmul(q_nope, uk_proj)  # (B, h, T, latent_dim)

        # score = q_nope * k_nope^T (absorption) + q_rope * k_rope^T (RoPE)
        attn_nope = torch.matmul(Q_absorbed, ckv.unsqueeze(-3).mT)  # (B, h, T, S)
        attn_rope = torch.matmul(q_rope, k_rope.unsqueeze(1).mT)    # (B, h, T, S), k_rope broadcast по головам
        attn_weights = (attn_nope + attn_rope) * self.scale
        if use_cache:
            q_pos = torch.arange(offset, S, device=device)
            k_pos = torch.arange(S, device=device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (T, S)
            attn_weights = attn_weights.masked_fill(~mask[None, None], float("-inf"))
        else:
            attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if use_cache:
            # absorption W_UV: P @ C @ W_UV^T == (P @ C) @ W_UV^T.
            # Применяем wuv к сжатому результату длины T, а не к V длины S.
            y_latent = torch.matmul(attn_weights, ckv.unsqueeze(1))  # (B, h, T, latent)
            wuv = self.wuv.weight.view(self.n_head, self.v_dim, self.latent_dim)
            y = torch.einsum("bhtl,hvl->bhtv", y_latent, wuv)
            y = y.transpose(1, 2).reshape(B, T, self.n_head * self.v_dim)
        else:
            v = self.wuv(ckv).view(B, T, self.n_head, self.v_dim).transpose(1, 2)
            y = torch.matmul(attn_weights, v)
            y = y.transpose(1, 2).reshape(B, T, self.n_head * self.v_dim)

        y = self.c_proj(y)
        if use_cache:
            self.cache_len = S

        return self.resid_dropout(y)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, device="cpu"):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.self_attn = CausalAttention(config, device)
        self.post_attention_layernorm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = Expert(config.n_embed, config.intermediate_size, bias=config.mlp_bias)
        else:
            self.mlp = DeepSeekMoE(config)

    def forward(self, x, use_cache=False, need_aux=False):
        x = x + self.self_attn(self.input_layernorm(x), use_cache=use_cache)
        h = self.post_attention_layernorm(x)

        if isinstance(self.mlp, DeepSeekMoE):
            out = self.mlp(h, return_aux=need_aux)
            h, aux = (out, None) if not need_aux else out
        else:
            h = self.mlp(h)
            aux = (
                {k: x.new_zeros(()) for k in ("exp_bal", "dev_bal", "comm_bal", "aux")}
                if need_aux else None
            )

        return x + h, aux


class DeepSeekV2(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i, device) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        for module in self.modules():
            self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def reset_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_cache()

    def forward(self, input_ids, targets=None, use_cache=False, return_aux=False):
        need_aux = return_aux or (targets is not None)

        if input_ids.is_floating_point() and input_ids.size(-1) == self.config.n_embed:
            x = input_ids
        else:
            x = self.embed_tokens(input_ids)

        aux_accum = None
        for layer in self.layers:
            x, aux = layer(x, use_cache=use_cache, need_aux=need_aux)
            if need_aux and aux is not None:
                if aux_accum is None:
                    aux_accum = dict(aux)
                else:
                    for k in aux_accum:
                        aux_accum[k] = aux_accum[k] + aux[k]

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            if aux_accum is not None:
                loss = loss + aux_accum["aux"]

        if return_aux:
            return logits, loss, aux_accum
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        self.reset_cache()
        ids = input_ids
        self.forward(ids, use_cache=True, return_aux=False)
        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids[:, -1:], use_cache=True, return_aux=False)
            if temperature != 1.0:
                logits = logits / temperature
            next_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        self.reset_cache()
        return ids


if __name__ == "__main__":
    def small_config(**kw):
        kwargs = dict(
            n_embed=32, n_head=4, latent_dim=16, q_lora_rank=32,
            qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
            block_size=16, max_batch_size=2, attn_pdrop=0.0, resid_pdrop=0.0,
            num_hidden_layers=3, first_k_dense_replace=1, intermediate_size=16,
            moe_intermediate_size=8, n_routed_experts=4, n_shared_experts=2,
            num_experts_per_tok=2, n_group=2, topk_group=1,
            routed_scaling_factor=1.0, topk_method="group_limited_greedy",
            vocab_size=64, initializer_range=0.02, rope_scaling_mscale=1.0,
            aux_loss_alpha=0.001, dev_loss_alpha=0.001, comm_loss_alpha=0.001,
        )
        kwargs.update(kw)
        return Config(**kwargs)

    torch.manual_seed(0)
    ids = torch.randint(0, 64, (1, 5))

    # 1) полный forward == пошаговая генерация с кэшем
    m = DeepSeekV2(small_config()).eval()
    full_logits, _ = m(ids, use_cache=False, return_aux=False)
    m.reset_cache()
    step_logits = []
    for t in range(ids.size(1)):
        logits, _ = m(ids[:, t:t + 1], use_cache=True, return_aux=False)
        step_logits.append(logits[:, -1])
    step_logits = torch.stack(step_logits, dim=1)
    err = (full_logits - step_logits).abs().max().item()
    print("cache equivalence max err:", err)
    assert err < 1e-5, err

    # 2) bf16
    m16 = DeepSeekV2(small_config()).to(torch.bfloat16).eval()
    logits16, _ = m16(ids, use_cache=False)
    print("bf16:", logits16.dtype, tuple(logits16.shape))
    assert logits16.dtype is torch.bfloat16
    m16.reset_cache()
    out16 = m16.generate(ids[:, :2], max_new_tokens=2)
    print("bf16 generate:", tuple(out16.shape), out16.dtype)
    assert out16.dtype is torch.int64

    # 3) backward через все параметры (все эксперты выбираются: top_k == n_experts)
    mb = DeepSeekV2(small_config(num_experts_per_tok=4, topk_group=2))
    targets = torch.randint(0, 64, (1, 5))
    logits, loss, aux = mb(ids, targets=targets, use_cache=False, return_aux=True)
    assert set(aux) == {"exp_bal", "dev_bal", "comm_bal", "aux"}
    loss.backward()
    missing = [n for n, p in mb.named_parameters() if p.grad is None]
    print("params without grad:", missing)
    assert not missing, missing
    print("aux:", {k: round(v.item(), 7) for k, v in aux.items()})
    print("\nALL CHECKS PASSED")
