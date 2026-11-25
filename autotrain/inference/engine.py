"""Incremental inference engine: bundle -> token-id generation with KV cache.

Design constraints (see INFERENCE.md):

* Read-only. The bundle is loaded through ``load_bundle`` with full contract
  verification (per-file checksum + per-tensor sha256 + tied alias). Nothing
  here writes to the bundle or to ``evals/``.
* No global RNG pollution. Sampling uses a local ``torch.Generator`` so a
  request can never disturb trainer/evaluator determinism.
* Learned positional embeddings. ``start_pos`` is the absolute position of
  the first token of a decode step; forgetting it silently corrupts output.
* CPU-first. This phone has no CUDA/MPS; thread count is capped explicitly.

The public surface is token ids, not text, because M1 has no tokenizer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from ..export.loader import BundleInfo, load_bundle
from ..trainer.model import WanxiangModel
from .tokenizer import validate_token_ids

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


def default_threads() -> int:
    raw = os.environ.get("WANXIANG_THREADS")
    n = int(raw) if raw else 4
    return max(1, min(n, os.cpu_count() or 1))


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    # Structural decoding controls. All default to OFF so existing behaviour
    # and every historical report stay byte-identical.
    repetition_penalty: float = 1.0        # >1 divides logits of seen tokens
    no_repeat_ngram_size: int = 0          # >0 bans tokens completing a seen n-gram
    frequency_penalty: float = 0.0         # extra penalty per prior occurrence

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.do_sample and self.temperature == 0:
            raise ValueError("temperature must be > 0 when do_sample is true")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")
        if self.frequency_penalty < 0:
            raise ValueError("frequency_penalty must be >= 0")
        if self.no_repeat_ngram_size == 1:
            raise ValueError("no_repeat_ngram_size=1 would ban all tokens")


@dataclass
class GenerationResult:
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # length | eos | stop | context_full
    elapsed_sec: float
    cache_bytes: int
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "finish_reason": self.finish_reason,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "cache_bytes": self.cache_bytes,
            "truncated": self.truncated,
        }


def _apply_repetition_controls(
    logits: torch.Tensor, cfg: GenerationConfig, context_ids: list[int]
) -> torch.Tensor:
    """Suppress loops before decoding, on the logits (so sampling benefits too).

    Three orthogonal, deterministic controls:

    * ``repetition_penalty`` (GPT-2 style): already-seen tokens have their
      logit divided by the penalty (or multiplied when negative).
    * ``frequency_penalty``: adds ``penalty * (count - 1)`` to the divisor,
      so a token repeated N times is pushed down progressively harder.
    * ``no_repeat_ngram_size``: bans any token that would complete an n-gram
      already present in the context (hard constraint, not a soft bias).

    All are no-ops at their defaults, keeping historical greedy behaviour
    byte-identical.
    """
    if (cfg.repetition_penalty == 1.0 and cfg.frequency_penalty == 0.0
            and cfg.no_repeat_ngram_size == 0):
        return logits
    out = logits.clone()
    vocab = out.numel()

    if cfg.repetition_penalty != 1.0 or cfg.frequency_penalty > 0.0:
        counts: dict[int, int] = {}
        for t in context_ids:
            counts[t] = counts.get(t, 0) + 1
        for tok, n in counts.items():
            if not 0 <= tok < vocab:
                continue
            divisor = cfg.repetition_penalty + cfg.frequency_penalty * (n - 1)
            if divisor <= 0:
                continue
            if out[tok] > 0:
                out[tok] = out[tok] / divisor
            else:
                out[tok] = out[tok] * divisor

    n = cfg.no_repeat_ngram_size
    if n > 1 and len(context_ids) >= n - 1:
        prefix = tuple(context_ids[-(n - 1):])
        banned: set[int] = set()
        for j in range(len(context_ids) - n + 1):
            if tuple(context_ids[j:j + n - 1]) == prefix:
                banned.add(context_ids[j + n - 1])
        for tok in banned:
            if 0 <= tok < vocab:
                out[tok] = float("-inf")
    return out


def _sample_next(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    generator: torch.Generator,
    context_ids: list[int] | None = None,
) -> int:
    """Greedy by default; temperature + top-k + top-p when sampling."""
    logits = _apply_repetition_controls(logits, cfg, context_ids or [])
    if not cfg.do_sample:
        return int(logits.argmax().item())

    probs = torch.softmax(logits.float().cpu() / cfg.temperature, dim=-1)
    if cfg.top_k > 0:
        k = min(cfg.top_k, probs.numel())
        vals, idx = torch.topk(probs, k)
        probs = torch.zeros_like(probs).scatter_(0, idx, vals)
    if cfg.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        # keep tokens whose cumulative mass (excluding self) is below top_p
        drop = (cumulative - sorted_probs) > cfg.top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    total = probs.sum()
    if total <= 0:
        return int(logits.argmax().item())
    probs = probs / total
    return int(torch.multinomial(probs, 1, generator=generator).item())


class InferenceEngine:
    """Single-model, single-sequence-at-a-time generator."""

    def __init__(
        self,
        model: WanxiangModel,
        *,
        vocab_size: int,
        max_seq_len: int,
        device: str = "cpu",
        threads: int | None = None,
        info: BundleInfo | None = None,
    ):
        self.device = device
        self.model = model.to(device).eval()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.info = info
        self.threads = threads or default_threads()
        torch.set_num_threads(self.threads)

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        device: str | None = None,
        threads: int | None = None,
        verify: bool = True,
    ) -> "InferenceEngine":
        model, info = load_bundle(bundle_dir, device=device, verify=verify)
        return cls(
            model,
            vocab_size=info.vocab_size,
            max_seq_len=info.max_seq_len,
            device=info.device,
            threads=threads,
            info=info,
        )

    # ------------------------------------------------------------------

    def cache_bytes(self, cached_tokens: int) -> int:
        """KV cache footprint: 2 (K,V) * layers * dim * tokens * dtype bytes."""
        dtype_bytes = torch.empty((), dtype=next(self.model.parameters()).dtype).element_size()
        return 2 * self.model.cfg.layers * self.model.cfg.dim * cached_tokens * dtype_bytes

    def model_id(self) -> str:
        if self.info is None:
            return "wanxiang-local"
        step = self.info.step
        return f"{self.info.bundle_dir.name}@step{step}"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_id(),
            "device": self.device,
            "torch": torch.__version__,
            "threads": self.threads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "layers": self.model.cfg.layers,
            "dim": self.model.cfg.dim,
            "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
            "cache_bytes_per_token": self.cache_bytes(1),
            "step": self.info.step if self.info else None,
            "tokenizer": "identity",
            "supports_text": False,
        }

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: list[int],
        cfg: GenerationConfig | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        """Prefill once, then decode one token at a time with a KV cache.

        ``on_token`` is called for every generated id (used for SSE streaming).
        """
        cfg = cfg or GenerationConfig()
        cfg.validate()
        ids = [int(t) for t in prompt_ids]
        validate_token_ids(ids, self.vocab_size)
        if len(ids) > self.max_seq_len:
            raise ValueError(
                f"prompt has {len(ids)} tokens but max_seq_len is {self.max_seq_len}"
            )

        budget = min(cfg.max_new_tokens, self.max_seq_len - len(ids))
        if budget < 1:
            raise ValueError("prompt already fills max_seq_len; no room to generate")
        truncated = budget < cfg.max_new_tokens

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.seed))
        started = time.time()
        generated: list[int] = []
        finish_reason = "context_full" if truncated else "length"

        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits, cache = self.model.forward_cached(x, None, 0)
            next_logits = logits[0, -1]

            for i in range(budget):
                token = _sample_next(next_logits, cfg, generator, ids + generated)
                generated.append(token)
                if on_token is not None:
                    on_token(token)
                if cfg.eos_token_id is not None and token == cfg.eos_token_id:
                    finish_reason = "eos"
                    break
                if token in cfg.stop_token_ids:
                    finish_reason = "stop"
                    break
                if i == budget - 1:
                    break
                step = torch.tensor([[token]], dtype=torch.long, device=self.device)
                # Absolute position of this token: prompt + tokens generated so far.
                logits, cache = self.model.forward_cached(step, cache, len(ids) + i)
                next_logits = logits[0, -1]

        return GenerationResult(
            token_ids=generated,
            prompt_tokens=len(ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            elapsed_sec=time.time() - started,
            cache_bytes=self.cache_bytes(len(ids) + len(generated)),
            truncated=truncated,
        )
