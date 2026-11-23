"""Frozen evaluation spec for wanxiang-eval-3 (word-level holdout).

All constants are sealed into eval3_hash. The holdout WORDS are also sealed
into the corpus manifest (see corpus.py), so a re-split invalidates reports.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..split import HOLDOUT_COUNT, HOLDOUT_SEED, MAX_ANSWER_CHARS

EVAL3_VERSION = "wanxiang-eval-3"
REPORT_VERSION = "wanxiang-eval3-report-1"

QA_COUNT = 150
GEN_TOKENS = 48             # answers <= 24 chars after cleaning; 48 leaves margin
CONTROL_ROTATION = 37
FROZEN = True

METRIC_WEIGHTS = {
    "char_f1": 0.40,
    "rouge_l": 0.30,
    "edit_similarity": 0.20,
    "exact_match": 0.10,
}
MIN_IMPROVEMENT = 0.02
QUALITY_THRESHOLD = 0.50
BASELINE_THRESHOLD = 0.05
CONTROL_TOLERANCE = 0.10


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def spec_payload(corpus_manifest_hash: str) -> dict[str, Any]:
    return {
        "evalspec_version": EVAL3_VERSION,
        "report_version": REPORT_VERSION,
        "corpus_manifest_hash": corpus_manifest_hash,
        "qa_count": QA_COUNT,
        "gen_tokens": GEN_TOKENS,
        "control_rotation": CONTROL_ROTATION,
        "holdout_seed": HOLDOUT_SEED,
        "holdout_count": HOLDOUT_COUNT,
        "max_answer_chars": MAX_ANSWER_CHARS,
        "metric_weights": METRIC_WEIGHTS,
        "min_improvement": MIN_IMPROVEMENT,
        "quality_threshold": QUALITY_THRESHOLD,
        "baseline_threshold": BASELINE_THRESHOLD,
        "control_tolerance": CONTROL_TOLERANCE,
        "judge": "none (objective character-level metrics only)",
    }


def eval3_hash(corpus_manifest_hash: str) -> str:
    return hashlib.sha256(
        _canonical(spec_payload(corpus_manifest_hash)).encode("utf-8")
    ).hexdigest()
