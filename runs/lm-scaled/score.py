"""Score the scaled (more-data, longer-training) model on the FROZEN eval.

Critical: every score here points at runs/corpus/lm (the original frozen
corpus). The expanded train stream is never read by eval_lm. Therefore the
eval_lm_hash MUST equal the baseline's 2b70b6f6793432ab, else the A/B is
invalid and compare_reports will refuse it.

Usage:
    python runs/lm-scaled/score.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"          # FROZEN — do not change
CONFIG = ROOT / "runs/lm-scaled/config.yaml"
RUN = ROOT / "runs/lm-scaled"
BASELINE_REPORT = ROOT / "runs/lm-baseline/reports/lm_trained.json"
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

    # Sort by the numeric step, NOT lexicographically: "ckpt_step_900.pt"
    # sorts after "ckpt_step_1200.pt" as a string and would silently score a
    # stale checkpoint.
    ckpts = sorted(
        (RUN / "checkpoints").glob("ckpt_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise RuntimeError("no checkpoints in runs/lm-scaled")
    final_ckpt = ckpts[-1]
    bundle = RUN / "bundle_scaled"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(final_ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])

    out = REPORTS / "lm_scaled.json"
    run_py([
        "-m", "autotrain.eval_lm", "score",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--out", str(out), "--seq-len", "128", "--batch-size", "16",
    ])
    cand = json.loads(out.read_text(encoding="utf-8"))["report"]

    # Guard: the candidate must be on the exact same frozen instrument.
    if cand["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError(
            f"eval_lm_hash drifted: {cand['eval_lm_hash']} != {EXPECTED_HASH}; "
            "A/B would be invalid — aborting"
        )

    base = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))["report"]
    cmp = run_py([
        "-m", "autotrain.eval_lm", "compare",
        "--baseline", str(BASELINE_REPORT), "--candidate", str(out),
    ])
    (REPORTS / "compare_step400_vs_scaled.json").write_text(cmp + "\n", encoding="utf-8")
    print(f"[baseline step400] ppl={base['scores']['perplexity']:.2f} ce={base['scores']['cross_entropy']:.4f}")
    print(f"[scaled {cand['bundle']['step']}] ppl={cand['scores']['perplexity']:.2f} ce={cand['scores']['cross_entropy']:.4f}")
    print("[compare]\n" + cmp)

    gen = run_py([
        "-m", "autotrain.eval_lm", "generate",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--prompt", "自然语言处理是", "--max-new-tokens", "48",
    ])
    (REPORTS / "continuation_scaled.json").write_text(gen + "\n", encoding="utf-8")
    print("[generate]\n" + gen)


if __name__ == "__main__":
    main()
