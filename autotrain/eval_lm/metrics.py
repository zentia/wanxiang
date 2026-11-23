"""Objective, deterministic token-level LM metrics.

No LLM-as-judge, no learned reward. Cross-entropy and perplexity are computed
over a frozen holdout token stream with the model in eval mode. Pure helpers
are separated so they can be unit-tested without a model.
"""

from __future__ import annotations

import math

import torch


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean next-token cross-entropy in nats, over all valid positions.

    ``logits`` is (B, T, V), ``targets`` is (B, T) with -1 for ignored
    positions. Returns a plain float; caller handles no_grad.
    """
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be (B,T,V), targets (B,T)")
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1)
    n_valid = int((flat_targets != -1).sum().item())
    if n_valid == 0:
        raise ValueError("no valid target tokens")
    loss = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, ignore_index=-1, reduction="sum"
    )
    return float(loss.item()) / n_valid


def perplexity(mean_cross_entropy: float) -> float:
    """exp(CE) in nats, with overflow guard."""
    if mean_cross_entropy < 0 or not math.isfinite(mean_cross_entropy):
        raise ValueError(f"invalid cross-entropy: {mean_cross_entropy}")
    if mean_cross_entropy > 700:
        return float("inf")
    return math.exp(mean_cross_entropy)


def bits_per_character(mean_cross_entropy: float, chars_per_token: float) -> float:
    """Bits/char = CE(nats/token) / (chars/token) / ln(2).

    Requires a chars-per-token ratio from the tokenizer; optional diagnostic.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return mean_cross_entropy / chars_per_token / math.log(2.0)


def aggregate_losses(total_nats: float, total_tokens: int) -> dict[str, float]:
    """Aggregate streaming loss sums into mean CE and perplexity."""
    if total_tokens < 1:
        raise ValueError("no tokens scored")
    ce = total_nats / total_tokens
    return {"cross_entropy": ce, "perplexity": perplexity(ce), "tokens": float(total_tokens)}
