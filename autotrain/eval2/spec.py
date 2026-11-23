"""Frozen evaluation spec for wanxiang-eval-2.

Every threshold, weight, window and seed is a module-level constant and is
hashed into ``evalspec_hash``. Changing any of them silently invalidates all
old reports (they become INCOMPARABLE), which is exactly the point.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EVAL2_VERSION = "wanxiang-eval-2"
REPORT_VERSION = "wanxiang-eval2-report-1"

# Frozen corpus construction
QA_COUNT = 150
GEN_TOKENS = 32
MIN_SOURCE_OFFSET = 800_000      # SFT token offset; far beyond training consumption
TRAIN_CONSUMED_MAX_TOKEN = 512_000
CONTROL_ROTATION = 37            # deterministic shuffled-reference negative control
FROZEN = True

# Objective quality metrics, all constants participate in the seal
METRIC_WEIGHTS = {
    "char_f1": 0.40,
    "rouge_l": 0.30,
    "edit_similarity": 0.20,
    "exact_match": 0.10,
}
MIN_IMPROVEMENT = 0.02           # composite delta needed for IMPROVED
QUALITY_THRESHOLD = 0.50         # composite needed for QUALITY_PASSED
BASELINE_THRESHOLD = 0.05        # below this -> BASELINE
CONTROL_TOLERANCE = 0.10         # control must not beat the real set by this much


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def spec_payload(corpus_manifest_hash: str) -> dict[str, Any]:
    return {
        "evalspec_version": EVAL2_VERSION,
        "report_version": REPORT_VERSION,
        "corpus_manifest_hash": corpus_manifest_hash,
        "qa_count": QA_COUNT,
        "gen_tokens": GEN_TOKENS,
        "min_source_offset": MIN_SOURCE_OFFSET,
        "train_consumed_max_token": TRAIN_CONSUMED_MAX_TOKEN,
        "control_rotation": CONTROL_ROTATION,
        "metric_weights": METRIC_WEIGHTS,
        "min_improvement": MIN_IMPROVEMENT,
        "quality_threshold": QUALITY_THRESHOLD,
        "baseline_threshold": BASELINE_THRESHOLD,
        "control_tolerance": CONTROL_TOLERANCE,
        "judge": "none (objective character-level metrics only)",
    }


def eval2_hash(corpus_manifest_hash: str) -> str:
    """Seal over the corpus + every constant. Different from M1.5 evalspec_hash."""
    return hashlib.sha256(
        _canonical(spec_payload(corpus_manifest_hash)).encode("utf-8")
    ).hexdigest()
