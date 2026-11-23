"""Frozen spec for eval-4 (retrieval usability). No learned judge."""

from __future__ import annotations

import hashlib
import json
from typing import Any

COMPARABLE_WITH_EVAL3 = False   # different question by design
EVAL4_VERSION = "wanxiang-eval-4"
REPORT_VERSION = "wanxiang-eval4-report-1"
INDEX_KEY = "retrieval"          # answer source that eval-4 scores
QUALITY_THRESHOLD = 0.90         # retrieval should be near-perfect on exact lookup

QA_COUNT = 150
MIN_DELTA = 0.10                 # retrieval vs generator is expected to be huge


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def spec_payload(corpus_manifest_hash: str, index_hash: str) -> dict[str, Any]:
    return {
        "evalspec_version": EVAL4_VERSION,
        "report_version": REPORT_VERSION,
        "question": "retrieval usability: can lookup answer the holdout QAs",
        "corpus_manifest_hash": corpus_manifest_hash,
        "index_hash": index_hash,
        "qa_count": QA_COUNT,
        "answer_source": INDEX_KEY,
        "comparable_with_eval3": COMPARABLE_WITH_EVAL3,
        "judge": "none (objective character metrics, identical to eval-2/3)",
        "note": "Index contains whole dictionary incl. holdout words; a high "
                "score is a lookup result, NOT a generalisation claim.",
    }


def eval4_hash(corpus_manifest_hash: str, index_hash: str) -> str:
    return hashlib.sha256(
        _canonical(spec_payload(corpus_manifest_hash, index_hash)).encode("utf-8")
    ).hexdigest()
