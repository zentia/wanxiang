"""Objective generation-quality scoring on frozen holdout prompts.

No LLM-as-judge, no learned reward, no semantic scoring. Every metric is a
deterministic function of the generated token-id sequence:

* ``distinct_1`` / ``distinct_2``  - unique unigram / bigram ratio (higher better)
* ``repeat_3gram_rate``            - fraction of 3-grams already seen earlier
* ``max_3gram_count``              - most repeated 3-gram (loop detector)
* ``max_repeat_run``               - longest run of an identical token
* ``loop_rate``                    - fraction of prompts that entered a loop
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..export.loader import load_bundle
from ..inference.engine import GenerationConfig, InferenceEngine

GEN_VERSION = "wanxiang-eval-gen-1"
EVAL_FAMILY = "wanxiang-eval-gen"
HOLDOUT_FILE = "holdout_tokens.bin"
DEFAULT_N_PROMPTS = 32
PROMPT_LEN = 64
GEN_LEN = 64
SEED = 0

# Loop = any 3-gram occurring at least this many times in one continuation.
LOOP_THRESHOLD = 3


class EvalGenError(ValueError):
    """Frozen ruler violation or corrupt corpus."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen prompt set.
# ---------------------------------------------------------------------------


def freeze_prompts(
    corpus_dir: str | Path,
    *,
    n: int = DEFAULT_N_PROMPTS,
    prompt_len: int = PROMPT_LEN,
) -> dict[str, Any]:
    """Deterministic, evenly-spaced prompt windows from the frozen LM holdout."""
    p = Path(corpus_dir)
    f = p / HOLDOUT_FILE
    if not f.is_file():
        raise EvalGenError(f"holdout token file not found: {f}")
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    arr = np.fromfile(f, dtype="<u2")
    usable = len(arr) - prompt_len - 1
    if usable < n:
        raise EvalGenError("holdout too short for the requested prompt windows")
    starts = [int(i * usable / n) for i in range(n)]
    prompts = [arr[s:s + prompt_len].astype(np.int64).tolist() for s in starts]
    payload = {
        "n": n,
        "prompt_len": prompt_len,
        "starts": starts,
        "prompts": prompts,
        "corpus_id": manifest["corpus_id"],
    }
    return {
        "prompts": prompts,
        "starts": starts,
        "sha256": _sha(_canonical(payload)),
        "corpus_id": manifest["corpus_id"],
    }


def build_spec(
    prompt_sha256: str, corpus_id: str, *, n: int, gen_len: int
) -> dict[str, Any]:
    doc = {
        "family": EVAL_FAMILY,
        "version": GEN_VERSION,
        "prompt_sha256": prompt_sha256,
        "corpus_id": corpus_id,
        "n_prompts": n,
        "prompt_len": PROMPT_LEN,
        "gen_len": gen_len,
        "seed": SEED,
        "decode": "greedy (do_sample=False)",
        "metric": "objective repetition statistics on frozen holdout prompts",
    }
    doc["gen_hash"] = _sha(_canonical(doc))[:16]
    return doc


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def sequence_metrics(ids: list[int]) -> dict[str, float]:
    """Pure function: repetition statistics of one generated continuation."""
    n = len(ids)
    if n == 0:
        return {
            "distinct_1": 0.0, "distinct_2": 0.0, "repeat_3gram_rate": 1.0,
            "max_3gram_count": 0.0, "max_repeat_run": 0.0, "looped": 1.0,
        }
    bigrams = [tuple(ids[i:i + 2]) for i in range(max(0, n - 1))]
    trigrams = [tuple(ids[i:i + 3]) for i in range(max(0, n - 2))]

    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for g in trigrams:
        if g in seen:
            repeated += 1
        seen.add(g)
    tri_counts = Counter(trigrams)
    max_tri = max(tri_counts.values(), default=0)

    max_run = 1
    cur = 1
    for a, b in zip(ids, ids[1:]):
        cur = cur + 1 if a == b else 1
        max_run = max(max_run, cur)

    return {
        "distinct_1": len(set(ids)) / n,
        "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
        "repeat_3gram_rate": repeated / max(1, len(trigrams)),
        "max_3gram_count": float(max_tri),
        "max_repeat_run": float(max_run),
        "looped": 1.0 if max_tri >= LOOP_THRESHOLD else 0.0,
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


# ---------------------------------------------------------------------------
# Score / compare.
# ---------------------------------------------------------------------------


def score(
    bundle_dir: str | Path,
    corpus_dir: str | Path,
    *,
    decode: dict[str, Any] | None = None,
    device: str = "cpu",
    threads: int = 4,
    n_prompts: int = DEFAULT_N_PROMPTS,
    gen_len: int = GEN_LEN,
) -> dict[str, Any]:
    decode = dict(decode or {})
    prompts = freeze_prompts(corpus_dir, n=n_prompts)
    model, info = load_bundle(bundle_dir, device=device)
    engine = InferenceEngine(
        model, vocab_size=info.vocab_size, max_seq_len=info.max_seq_len,
        device=device, threads=threads, info=info,
    )
    cfg = GenerationConfig(
        max_new_tokens=gen_len, do_sample=False, seed=SEED, **decode
    )
    cfg.validate()

    rows: list[dict[str, float]] = []
    samples: list[list[int]] = []
    for prompt in prompts["prompts"]:
        result = engine.generate(list(prompt), cfg)
        rows.append(sequence_metrics(result.token_ids))
        if len(samples) < 2:
            samples.append(result.token_ids)

    spec = build_spec(prompts["sha256"], prompts["corpus_id"],
                      n=n_prompts, gen_len=gen_len)
    return {
        "status": "OK",
        "version": GEN_VERSION,
        "gen_hash": spec["gen_hash"],
        "spec": spec,
        "bundle_step": info.step,
        "decode_config": {
            "repetition_penalty": cfg.repetition_penalty,
            "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
            "frequency_penalty": cfg.frequency_penalty,
            "do_sample": cfg.do_sample,
        },
        "metrics": _aggregate(rows),
        "n_prompts": n_prompts,
        "sample_token_ids": samples,
        "note": "Objective repetition stats on frozen holdout prompts. "
                "Decode controls are the variable under test and are NOT in gen_hash.",
    }


# Higher is better for these; lower is better for the rest.
_HIGHER_BETTER = ("distinct_1", "distinct_2")
_LOWER_BETTER = ("repeat_3gram_rate", "max_3gram_count",
                 "max_repeat_run", "looped")


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A/B on one frozen ruler. Different gen_hash -> INCOMPARABLE."""
    if before.get("gen_hash") != after.get("gen_hash"):
        return {
            "verdict": "INCOMPARABLE",
            "reason": "different gen_hash (prompt set / lengths / seed differ)",
            "before_hash": before.get("gen_hash"),
            "after_hash": after.get("gen_hash"),
        }
    b, a = before["metrics"], after["metrics"]
    deltas = {k: a[k] - b[k] for k in _HIGHER_BETTER + _LOWER_BETTER}
    improved = sum(
        (deltas[k] > 0 if k in _HIGHER_BETTER else deltas[k] < 0)
        for k in deltas
    )
    regressed = sum(
        (deltas[k] < 0 if k in _HIGHER_BETTER else deltas[k] > 0)
        for k in deltas
    )
    verdict = "IMPROVED" if improved > regressed else (
        "REGRESSED" if regressed > improved else "NO_CHANGE"
    )
    return {
        "verdict": verdict,
        "gen_hash": before["gen_hash"],
        "before_step": before.get("bundle_step"),
        "after_step": after.get("bundle_step"),
        "before_decode": before.get("decode_config"),
        "after_decode": after.get("decode_config"),
        "metrics_before": b,
        "metrics_after": a,
        "deltas": deltas,
        "improved_metrics": improved,
        "regressed_metrics": regressed,
    }
