"""Minimal GPT-style decoder-only transformer (Wanxiang base model).

Kept deliberately small and readable: pre-norm blocks, causal
self-attention, SwiGLU MLP, RMSNorm, learned positional embeddings.
~100M class models are the M1 target scale.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ExperimentConfig, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegating keeps the training/eval path numerically identical by
        # construction: forward_cached(..., past=None) runs the same ops.
        return self.forward_cached(x, None)[0]

    def forward_cached(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attention with an explicit KV cache.

        ``past`` is (k, v) with shape (B, heads, T_past, head_dim). Prefill
        passes ``past=None`` (causal mask, q_len == k_len). Decode passes the
        cache and drops the causal mask: every cached key is a visible past
        token, so no masking is needed. Returns (output, new_cache).
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if past is not None:
            pk, pv = past
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3):
        super().__init__()
        hidden = int(dim * hidden_mult) // 64 * 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cached(x, None)[0]

    def forward_cached(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        a, new_past = self.attn.forward_cached(self.norm1(x), past)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_past


class WanxiangModel(nn.Module):
    """Decoder-only transformer with learned positional embeddings."""

    def __init__(self, cfg: ModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(max_seq_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits, _ = self._forward_impl(idx, kv_cache=None, start_pos=0)
        if targets is None:
            return logits, None
        flat_targets = targets.reshape(-1)
        n_valid = int((flat_targets != -1).sum().item())
        if n_valid == 0:
            # Answer-only loss: a window can contain no answer tokens. Return a
            # differentiable zero instead of NaN (which would poison backward).
            return logits, logits.sum() * 0.0
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            flat_targets,
            ignore_index=-1,
        )
        return logits, loss

    def forward_cached(
        self,
        idx: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Incremental forward. Returns (logits for these tokens, new cache).

        ``start_pos`` is the absolute position of ``idx[:, 0]`` — required
        because this model uses *learned* positional embeddings, not RoPE.
        Prefill: kv_cache=None / [] and start_pos=0. Decode: pass the cache
        returned by the previous call and start_pos = number of cached tokens.
        """
        return self._forward_impl(idx, kv_cache=kv_cache, start_pos=start_pos)

    def _forward_impl(
        self,
        idx: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape
        if start_pos + T > self.max_seq_len:
            raise ValueError(
                f"position {start_pos + T} exceeds max_seq_len {self.max_seq_len}"
            )
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            x, layer_cache = block.forward_cached(x, past)
            new_cache.append(layer_cache)
        x = self.norm(x)
        return self.lm_head(x), new_cache

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n


def seed_everything(seed: int) -> None:
    """Seed every RNG a run consumes so a fresh start is reproducible.

    Checkpoints restore python/numpy/torch/cuda states on resume, so this
    only affects runs that start from step 0. ``torch.manual_seed`` also
    seeds CUDA when available.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _ = torch.manual_seed(seed)


def init_model(cfg: ExperimentConfig, seed: int | None = None) -> WanxiangModel:
    """Single construction path shared by training and the step-0 baseline.

    Seeding immediately before construction is what makes a fresh
    ``TrainerRunner`` and ``export_initial`` produce bitwise-identical
    weights. Changing that order, or constructing the model separately in
    either place, silently breaks the A/B "before" arm -- so both callers
    must go through this function.
    """
    seed_everything(int(cfg.seed if seed is None else seed))
    return WanxiangModel(cfg.model, max_seq_len=cfg.data.seq_len)


def estimate_flops_per_token(cfg: ModelConfig) -> float:
    """Rough forward-pass FLOPs per token: ~6*N*d rule for training is
    6*N total (fwd+bwd); here we return forward-only 2*N."""
    n = cfg.layers * (
        12 * cfg.dim * cfg.dim  # attention qkv+proj
        + 3 * cfg.dim * int(cfg.dim * 8 / 3)  # swiglu
    )
    return 2.0 * n + 2.0 * cfg.dim * cfg.vocab_size
