"""eval-lm runner: perplexity + deterministic continuation from a sealed bundle.

Loads a model via the standard bundle loader (no training-loop dependency),
scores the frozen holdout token stream, and emits a report whose eval_lm_hash
binds it to the exact corpus/holdout/tokenizer/config. Comparison across
families or hashes is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..export.contract import sha256_file
from ..export.loader import load_bundle
from ..inference.tokenizer import load_tokenizer
from .metrics import aggregate_losses
from .spec import EVAL_FAMILY, build_spec

HOLDOUT_FILE = "holdout_tokens.bin"
MANIFEST_FILE = "manifest.json"
TOKENIZER_FILE = "tokenizer.json"


class EvalLMError(ValueError):
    """Raised when eval-lm inputs are inconsistent or comparison is invalid."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _load_holdout(corpus_dir: str | Path) -> np.ndarray:
    p = Path(corpus_dir) / HOLDOUT_FILE
    if not p.is_file():
        raise EvalLMError(f"holdout token file not found: {p}")
    arr = np.fromfile(p, dtype="<u2")
    if arr.size < 64:
        raise EvalLMError("holdout token stream too short")
    return arr


def score_bundle(
    bundle_dir: str | Path,
    corpus_dir: str | Path,
    *,
    seq_len: int = 128,
    batch_size: int = 16,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Score a model bundle on the frozen LM holdout; return a sealed report."""
    corpus = Path(corpus_dir)
    manifest_path = corpus / MANIFEST_FILE
    if not manifest_path.is_file():
        raise EvalLMError(f"corpus manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout_path = corpus / HOLDOUT_FILE
    holdout_sha = sha256_file(holdout_path)
    if holdout_sha != manifest["files"][HOLDOUT_FILE]["sha256"]:
        raise EvalLMError("holdout checksum mismatch (corpus was modified)")

    tok = load_tokenizer(corpus / TOKENIZER_FILE)
    if tok.as_dict()["sha256"] != manifest["tokenizer"]["sha256"]:
        raise EvalLMError("tokenizer checksum mismatch")

    model, info = load_bundle(bundle_dir, device=device)
    if model.max_seq_len < seq_len:
        raise EvalLMError(
            f"bundle max_seq_len {model.max_seq_len} < eval seq_len {seq_len}"
        )

    spec = build_spec(
        corpus_id=manifest["corpus_id"],
        corpus_version=manifest["version"],
        holdout_sha256=holdout_sha,
        holdout_tokens=int(manifest["files"][HOLDOUT_FILE]["tokens"]),
        tokenizer_sha256=manifest["tokenizer"]["sha256"],
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )

    arr = _load_holdout(corpus)
    # Non-overlapping windows of (seq_len + 1) tokens: seq_len inputs, seq_len targets.
    span = seq_len + 1
    n_windows = arr.size // span
    if n_windows < 1:
        raise EvalLMError("holdout shorter than one eval window")
    x = torch.from_numpy(arr[: n_windows * span].reshape(n_windows, span).astype(np.int64))
    xb, yb = x[:, :-1], x[:, 1:]

    total_nats = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            x_chunk = xb[start:start + batch_size].to(info.device)
            y_chunk = yb[start:start + batch_size].to(info.device)
            logits, _ = model(x_chunk)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_targets = y_chunk.reshape(-1)
            loss_sum = torch.nn.functional.cross_entropy(
                flat_logits, flat_targets, reduction="sum"
            )
            total_nats += float(loss_sum.item())
            total_tokens += int(flat_targets.numel())

    scores = aggregate_losses(total_nats, total_tokens)
    report: dict[str, Any] = {
        "family": EVAL_FAMILY,
        "version": spec["version"],
        "eval_lm_hash": spec["eval_lm_hash"],
        "spec": spec,
        "bundle": {
            "path": str(Path(bundle_dir).resolve()),
            "step": info.step,
            "model_id": info.config.get("architecture"),
            "dtype": info.dtype,
            "device": info.device,
        },
        "corpus": {
            "path": str(corpus.resolve()),
            "corpus_id": manifest["corpus_id"],
        },
        "scores": scores,
        "windows_scored": n_windows,
        "comparability": {
            "comparable_with": [EVAL_FAMILY],
            "incomparable_with": list(spec["incomparable_with"]),
            "note": "Cross-entropy/perplexity on a frozen text holdout. "
                    "NOT comparable with eval2/3/4 answer-quality or eval4 retrieval.",
        },
    }
    return report


def generate_continuation(
    bundle_dir: str | Path,
    corpus_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 48,
    device: str | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministic greedy continuation from a text prompt (qualitative aid).

    Diagnostics only: continuation text is NOT a scored metric and cannot be
    compared across bundles with the strict perplexity comparator.

    ``tokenizer_path`` may override the corpus tokenizer for models trained
    with a different-but-id-compatible tokenizer (e.g. char-id-aligned BPE).
    Scoring never uses this override: ``score_bundle`` always keeps the frozen
    corpus tokenizer so eval_lm_hash cannot drift.
    """
    corpus = Path(corpus_dir)
    tok_path = Path(tokenizer_path) if tokenizer_path is not None else corpus / TOKENIZER_FILE
    tok = load_tokenizer(tok_path)
    if not tok.supports_text:
        raise EvalLMError("tokenizer does not support text")
    model, info = load_bundle(bundle_dir, device=device)
    ids = tok.encode(prompt, add_bos=True)
    if not ids:
        raise EvalLMError("empty prompt after encoding")
    if len(ids) > model.max_seq_len:
        ids = ids[-model.max_seq_len:]
    generated = generate_tokens(
        model, ids, max_new_tokens=max_new_tokens, device=info.device,
        eos_token_id=tok.eos_token_id,
    )
    return {
        "prompt": prompt,
        "continuation": tok.decode(generated[len(ids):]),
        "max_new_tokens": int(max_new_tokens),
        "note": "greedy, deterministic; diagnostic only, not a scored metric",
    }


def generate_tokens(
    model: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: str,
    eos_token_id: int | None = None,
) -> list[int]:
    """Greedy token generation with the model's KV cache (eval mode).

    Prefills the whole prompt in one forward pass, then decodes one token at
    a time. Stops on EOS or when the learned position limit is reached.
    """
    model.eval()
    ids = list(prompt_ids)
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    pos = 0
    with torch.no_grad():
        # prefill
        chunk = torch.tensor([ids], dtype=torch.long, device=device)
        logits, kv_cache = model.forward_cached(chunk, kv_cache, start_pos=0)
        pos = len(ids)
        next_id = int(logits[0, -1].argmax().item())
        for _ in range(max_new_tokens):
            if eos_token_id is not None and next_id == eos_token_id:
                break
            if pos >= model.max_seq_len:
                break
            ids.append(next_id)
            step = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, kv_cache = model.forward_cached(step, kv_cache, start_pos=pos)
            pos += 1
            next_id = int(logits[0, -1].argmax().item())
    return ids


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], *, threshold: float = 0.0
) -> dict[str, Any]:
    """Strict LM comparison: same family AND same eval_lm_hash required.

    Lower perplexity is better. Returns a verdict; never compares across
    corpus/holdout/tokenizer/config. ``threshold`` is the minimum absolute
    perplexity improvement required to call it IMPROVED.
    """
    if baseline.get("family") != EVAL_FAMILY or candidate.get("family") != EVAL_FAMILY:
        raise EvalLMError("both reports must be eval-lm family")
    hb, hc = baseline.get("eval_lm_hash"), candidate.get("eval_lm_hash")
    if hb != hc:
        raise EvalLMError(
            f"refusing to compare different eval_lm_hash ({hb} vs {hc}); "
            "corpus/holdout/tokenizer/config must be identical"
        )
    if threshold < 0:
        raise EvalLMError("threshold must be >= 0")
    base_ppl = baseline["scores"]["perplexity"]
    cand_ppl = candidate["scores"]["perplexity"]
    delta = base_ppl - cand_ppl  # positive = candidate perplexity is lower = better
    verdict = "IMPROVED" if delta > threshold else "NO_CHANGE"
    return {
        "comparable": True,
        "eval_lm_hash": hb,
        "baseline_perplexity": base_ppl,
        "candidate_perplexity": cand_ppl,
        "perplexity_delta": delta,
        "baseline_cross_entropy": baseline["scores"]["cross_entropy"],
        "candidate_cross_entropy": candidate["scores"]["cross_entropy"],
        "threshold": threshold,
        "verdict": verdict,
    }


def run(
    bundles: list[tuple[str, str | Path]],
    corpus_dir: str | Path,
    out_path: str | Path,
    *,
    seq_len: int = 128,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score several named bundles on one frozen holdout and write a report."""
    reports: dict[str, Any] = {}
    for name, bundle in bundles:
        reports[name] = score_bundle(
            bundle, corpus_dir, seq_len=seq_len, batch_size=batch_size
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"eval": "eval-lm", "reports": reports}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return doc
