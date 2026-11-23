"""eval-lm frozen spec and hash namespace.

The eval family is part of the hash so a report from one instrument can never
be compared with another. ``compare_reports`` refuses to compare across
families/hashes, preventing a perplexity gain (or a retrieval lookup) from
being presented as progress on a different task.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EVAL_FAMILY = "wanxiang-eval-lm"
EVAL_VERSION = "wanxiang-eval-lm-1"
# Instruments that are explicitly NOT commensurable with this one.
INCOMPARABLE_WITH = ("wanxiang-eval-2", "wanxiang-eval-3", "wanxiang-eval-4")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def build_spec(
    *,
    corpus_id: str,
    corpus_version: str,
    holdout_sha256: str,
    holdout_tokens: int,
    tokenizer_sha256: str,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "family": EVAL_FAMILY,
        "version": EVAL_VERSION,
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "holdout_sha256": holdout_sha256,
        "holdout_tokens": int(holdout_tokens),
        "tokenizer_sha256": tokenizer_sha256,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "metric": "token-level cross-entropy / perplexity over frozen holdout",
        "comparable_with": [EVAL_FAMILY],
        "incomparable_with": list(INCOMPARABLE_WITH),
    }
    doc["eval_lm_hash"] = hashlib.sha256(_canonical(doc).encode("utf-8")).hexdigest()[:16]
    return doc
