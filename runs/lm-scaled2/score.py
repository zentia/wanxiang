"""Score the ext2 + 2400-step model on the FROZEN eval_lm instrument.

Candidate is compared against runs/lm-scaled step1200, so the only changes are
more disjoint train documents and a longer step budget. The eval corpus,
holdout, tokenizer and hash stay byte-identical (2b70b6f6793432ab).

Usage:
    python runs/lm-scaled2/score.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"          # FROZEN — do not change
CONFIG = ROOT / "runs/lm-scaled2/config.yaml"
RUN = ROOT / "runs/lm-scaled2"
REF_REPORT = ROOT / "runs/lm-scaled/reports/lm_scaled.json"
EXPECTED_HASH = "2b70b6f6793432ab"
REPORTS = RUN / "reports"
PY = ROOT / ".venv/bin/python"


def run_py(args: list[str]) -> str:
    out = subprocess.run(
        [str(PY), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(args)}\n{out.stderr[-2000:]}")
    return out.stdout


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        (RUN / "checkpoints").glob("ckpt_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise RuntimeError("no checkpoints in runs/lm-scaled2")
    final_ckpt = ckpts[-1]
    bundle = RUN / "bundle_scaled2"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(final_ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])

    out = REPORTS / "lm_scaled2.json"
    run_py([
        "-m", "autotrain.eval_lm", "score",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--out", str(out), "--seq-len", "128", "--batch-size", "16",
    ])
    cand = json.loads(out.read_text(encoding="utf-8"))["report"]
    if cand["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError(
            f"eval_lm_hash drifted: {cand['eval_lm_hash']} != {EXPECTED_HASH}; "
            "A/B invalid — aborting"
        )

    ref = json.loads(REF_REPORT.read_text(encoding="utf-8"))["report"]
    cmp = run_py([
        "-m", "autotrain.eval_lm", "compare",
        "--baseline", str(REF_REPORT), "--candidate", str(out),
    ])
    (REPORTS / "compare_step1200_vs_scaled2.json").write_text(cmp + "\n", encoding="utf-8")
    print(f"[ref step1200] ppl={ref['scores']['perplexity']:.2f} ce={ref['scores']['cross_entropy']:.4f}")
    print(f"[candidate {cand['bundle']['step']}] ppl={cand['scores']['perplexity']:.2f} ce={cand['scores']['cross_entropy']:.4f}")
    print("[compare]\n" + cmp)

    gen = run_py([
        "-m", "autotrain.eval_lm", "generate",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--prompt", "自然语言处理是", "--max-new-tokens", "48",
    ])
    (REPORTS / "continuation_scaled2.json").write_text(gen + "\n", encoding="utf-8")
    print("[generate]\n" + gen)


if __name__ == "__main__":
    main()
