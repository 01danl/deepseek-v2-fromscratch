import argparse
import math
import os
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import Config, DeepSeekV2


def make_config(**overrides):
    kwargs = dict(
        vocab_size=12000,
        tie_word_embeddings=True,
        n_embed=384,
        n_head=6,
        num_hidden_layers=8,
        first_k_dense_replace=1,
        intermediate_size=1024,
        block_size=1024,
        latent_dim=128,
        q_lora_rank=None,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=12,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        n_group=3,
        topk_group=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


class ByteBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {bytes([i]): i for i in range(256)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []

    def train(self, data, max_tokens=None):
        tokens = list(data[:max_tokens]) if max_tokens else list(data)
        banned = set()
        while len(self.vocab) < self.vocab_size:
            pairs = Counter(zip(tokens, tokens[1:]))
            pair = None
            for candidate, _ in pairs.most_common():
                if candidate in banned:
                    continue
                merged = self.id_to_bytes[candidate[0]] + self.id_to_bytes[candidate[1]]
                if merged not in self.vocab:
                    pair = candidate
                    break
            if pair is None:
                break

            merged = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            new_id = len(self.vocab)
            self.vocab[merged] = new_id
            self.id_to_bytes[new_id] = merged
            self.merges.append((pair, new_id))

            out = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def encode(self, data):
        tokens = list(data)
        for pair, new_id in self.merges:
            out = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def decode(self, tokens):
        return b"".join(self.id_to_bytes[t] for t in tokens)

    def state_dict(self):
        return {"vocab_size": self.vocab_size, "merges": self.merges}


def get_batch(tokens, B, T, device):
    n = B * T
    ix = torch.randint(len(tokens) - n - 1, (n,))
    x = torch.tensor([tokens[i] for i in ix], dtype=torch.long).view(B, T)
    y = torch.tensor([tokens[i + 1] for i in ix], dtype=torch.long).view(B, T)
    return x.to(device), y.to(device)


def validate_config(config):
    assert config.n_routed_experts % config.n_group == 0
    per_group = config.n_routed_experts // config.n_group
    assert config.topk_group * per_group >= config.num_experts_per_tok, (
        "topk_group * experts_per_group must be >= num_experts_per_tok"
    )


def count_params(config, model):
    total = sum(p.numel() for p in model.parameters())
    inactive_per_layer = (config.n_routed_experts - config.num_experts_per_tok) * (
        3 * config.n_embed * config.moe_intermediate_size
    )
    num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
    active = total - num_moe_layers * inactive_per_layer
    return total, active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="путь к текстовому файлу")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=12000)
    p.add_argument("--max-train-bytes", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--sample-tokens", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = make_config(vocab_size=args.vocab_size, block_size=args.block_size)
    validate_config(config)

    tokenizer = ByteBPE(config.vocab_size)
    data = open(args.data, "rb").read()
    tokens = tokenizer.train(data, max_tokens=args.max_train_bytes)
    if len(tokens) <= args.batch_size * args.block_size:
        raise ValueError("corpus too short for the requested batch_size * block_size")
    print(f"tokens: {len(tokens)} | vocab: {len(tokenizer.vocab)}")

    model = DeepSeekV2(config, device).to(device)
    total, active = count_params(config, model)
    print(f"Total:  {total / 1e6:.2f}M")
    print(f"Active: {active / 1e6:.2f}M")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    warmup = max(1, int(args.steps * args.warmup_ratio))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and not args.no_amp
        else nullcontext()
    )

    for step in range(args.steps):
        x, y = get_batch(tokens, args.batch_size, args.block_size, device)
        model.train()
        model.reset_cache()
        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits, loss, aux = model(x, targets=y, use_cache=False, return_aux=True)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            ce = loss.detach() - aux["aux"].detach()
            print(
                f"step {step:5d} | ce {ce.item():.4f} | aux {aux['aux'].item():.6f} "
                f"| exp {aux['exp_bal'].item():.6f} dev {aux['dev_bal'].item():.6f} "
                f"comm {aux['comm_bal'].item():.6f} | lr {scheduler.get_last_lr()[0]:.2e} "
                f"| grad {grad_norm.item():.3f}"
            )

        if args.eval_every and (step + 1) % args.eval_every == 0:
            model.eval()
            model.reset_cache()
            prompt = x[:1, :min(args.prompt_len, x.size(1))]
            with torch.no_grad():
                sample = model.generate(
                    prompt, max_new_tokens=args.sample_tokens, temperature=0, top_k=args.top_k
                )[0].tolist()
            print("sample:", repr(tokenizer.decode(sample)[: args.sample_tokens * 4]))
            model.train()

        if args.save_every and (step + 1) % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "tokenizer": tokenizer.state_dict(),
                },
                args.out,
            )
            print("saved:", args.out)


if __name__ == "__main__":
    main()
