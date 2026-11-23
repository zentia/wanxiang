"""Score retrieval vs generator on the SAME frozen eval-3 QA set.

    python runs/eval4-retrieval/eval_all.py

Reads:
  - runs/eval4-retrieval/index          (sealed dictionary index)
  - evals3/qa.jsonl                     (frozen questions + references, untouched)
  - runs/eval3-clean/reports/eval3_new_step1350.json  (char 10M generator, same QA)
  - runs/eval3-large/reports/eval3_large_bundle_step_1300.json (100M generator)
Writes runs/eval4-retrieval/reports/.

Honesty: eval-4 is INCOMPARABLE with eval-3 as a training verdict. Here we only
reuse the same questions/references to contrast two answer sources.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.eval4 import (Eval4Runner, compare_eval4, eval4_hash)  # noqa: E402
from autotrain.eval3.corpus import verify_corpus  # noqa: E402
from autotrain.retrieval import Retriever, verify_index  # noqa: E402

INDEX = ROOT / "runs/eval4-retrieval/index"
REPORTS = ROOT / "runs/eval4-retrieval/reports"
GEN_CHAR = ROOT / "runs/eval3-clean/reports/eval3_new_step1350.json"
GEN_LARGE = ROOT / "runs/eval3-large/reports/eval3_large_bundle_step_1300.json"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    idx = verify_index(INDEX)
    corpus = verify_corpus(ROOT / "evals3")
    retriever = Retriever.from_dir(INDEX)

    report = Eval4Runner(
        retriever, eval_dir=ROOT / "evals3", index_hash=idx["index_id"]
    ).run()
    assert report["comparable_with_eval3"] is False
    assert report["corpus_manifest_hash"] == corpus["corpus_manifest_hash"]
    assert report["evalspec_hash"] == eval4_hash(
        corpus["corpus_manifest_hash"], idx["index_id"]
    )
    (REPORTS / "eval4_retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparisons = {}
    for name, path in (("char10m_step1350", GEN_CHAR), ("large100m_step1300", GEN_LARGE)):
        if not path.is_file():
            continue
        gen = json.loads(path.read_text(encoding="utf-8"))
        cmp = compare_eval4(gen, report)
        comparisons[name] = cmp
        (REPORTS / f"compare_{name}.json").write_text(
            json.dumps(cmp, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("EVAL4_OK=" + json.dumps({
        "retrieval_composite": round(report["metrics"]["composite"], 4),
        "retrieval_exact_match": report["metrics"]["exact_match"],
        "exact_hit_rate": round(report["exact_hit_rate"], 4),
        "methods": report["methods"],
        "comparisons": {k: round(v.get("delta", 0), 4) for k, v in comparisons.items()},
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
