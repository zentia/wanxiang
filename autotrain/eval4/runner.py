"""eval-4 runner: score retrieval answers on the frozen eval-3 QA questions.

It reads the SAME evals3/qa.jsonl questions/references (so scoring is against
byte-identical references) but does NOT share eval-3's verdict: eval-4 is a
usability report for the retrieval tool, explicitly incomparable with the
zero-shot generator report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..eval2.metrics import aggregate, score_pair
from ..eval3.corpus import read_qa, verify_corpus
from ..retrieval import Retriever
from .spec import (COMPARABLE_WITH_EVAL3, EVAL4_VERSION, INDEX_KEY,
                   QUALITY_THRESHOLD, REPORT_VERSION, eval4_hash)


def run_retrieval(
    retriever: Retriever,
    *,
    eval_dir: str | Path | None = None,
    index_hash: str = "",
) -> dict[str, Any]:
    """Score retrieval over the frozen QA set. Pure, no model/network."""
    manifest = verify_corpus(eval_dir)
    qa, _control = read_qa(eval_dir)
    corpus_hash = manifest["corpus_manifest_hash"]

    replies: list[str] = []
    methods: list[str] = []
    scores: list[dict[str, float]] = []
    examples: list[dict[str, Any]] = []
    for row in qa:
        hit = retriever.best(row["question"])
        reply = hit.answer if hit and hit.method == "exact" else ""
        method = hit.method if hit else "none"
        replies.append(reply)
        methods.append(method)
        s = score_pair(reply, row["reference"])
        scores.append(s)
        if len(examples) < 5:
            examples.append({
                "question": row["question"],
                "reference": row["reference"],
                "reply": reply,
                "method": method,
                "composite": round(s["composite"], 4),
            })

    metrics = aggregate(scores)
    exact_hits = sum(1 for m in methods if m == "exact")
    report: dict[str, Any] = {
        "format_version": REPORT_VERSION,
        "evalspec_version": EVAL4_VERSION,
        "evalspec_hash": eval4_hash(corpus_hash, index_hash),
        "corpus_manifest_hash": corpus_hash,
        "index_hash": index_hash,
        "comparable_with_eval3": COMPARABLE_WITH_EVAL3,
        "qa_count": len(qa),
        "answer_source": INDEX_KEY,
        "metrics": metrics,
        "exact_hit_rate": exact_hits / max(1, len(qa)),
        "methods": {m: methods.count(m) for m in sorted(set(methods))},
        "checks": {
            "quality_passed": metrics["composite"] >= QUALITY_THRESHOLD,
        },
        "status": "PASSED" if metrics["composite"] >= QUALITY_THRESHOLD else "PARTIAL",
        "examples": examples,
        "note": "Lookup tool with full dictionary (holdout included). High "
                "score = retrieval works, NOT that the generator generalised.",
    }
    return report


class Eval4Runner:
    """Convenience wrapper mirroring Eval3Runner's construction shape."""

    def __init__(self, retriever: Retriever, *, eval_dir: str | Path | None = None,
                 index_hash: str = ""):
        self.retriever = retriever
        self.eval_dir = Path(eval_dir) if eval_dir is not None else None
        self.index_hash = index_hash

    def run(self) -> dict[str, Any]:
        return run_retrieval(self.retriever, eval_dir=self.eval_dir,
                             index_hash=self.index_hash)


def compare_eval4(generator: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    """Compare generator vs retrieval on the SAME frozen QA set (honest A/B).

    Requires both reports to reference the same eval-3 corpus manifest so the
    questions and references are byte-identical. The generator report uses
    eval-3 metrics (same weights), so composite is directly comparable here.
    """
    diffs = []
    if generator.get("corpus_manifest_hash") != retrieval.get("corpus_manifest_hash"):
        diffs.append("corpus_manifest_hash")
    if diffs:
        return {
            "format_version": "wanxiang-eval4-compare-1",
            "compatible": False,
            "differences": diffs,
            "verdict": "INCOMPARABLE",
        }
    g = generator["metrics"]["composite"]
    r = retrieval["metrics"]["composite"]
    return {
        "format_version": "wanxiang-eval4-compare-1",
        "compatible": True,
        "generator_composite": g,
        "retrieval_composite": r,
        "delta": r - g,
        "generator_exact_match": generator["metrics"]["exact_match"],
        "retrieval_exact_match": retrieval["metrics"]["exact_match"],
        "verdict": "RETRIEVAL_SUPERIOR" if r > g else "NO_GAIN",
        "note": "Same questions/references; answer source differs "
                "(generator vs lookup). This measures tool usability.",
    }
