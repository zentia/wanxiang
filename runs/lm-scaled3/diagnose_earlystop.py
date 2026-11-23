"""Diagnostic: was the step-4350 early stop premature on the FROZEN ruler?

The training-internal early stop fired at step 4350 because the tail-val loss
had not improved since step 3750 (12 evaluations). That val slice is tiny
(16,384 tokens of a 312,197-token tail), contiguous, and swaps out whenever
the train stream grows -- so it is not a trustworthy cross-run convergence
signal.

This script scores a dense set of checkpoints across the early-stop window on
the FROZEN eval_lm instrument (eval_lm_hash=2b70b6f6793432ab) and reports the
perplexity curve. If perplexity keeps dropping after step 3750, the early stop
was premature with respect to the metric we actually care about.

Self-contained: does not import or modify the versioned score.py.

Usage:
    python runs/lm-scaled3/diagnose_earlystop.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"          # FROZEN
CONFIG = ROOT / "runs/lm-scaled3/config.yaml"
RUN = ROOT / "runs/lm-scaled3"
EXPECTED_HASH = "2b70b6f6793432ab"
DIAG = RUN / "reports"
PY = ROOT / ".venv/bin/python"

# All 100-step checkpoints in the early-stop window. 3600/4000/4350 were
# already scored by score.py; we re-score them here for one uniform table.
STEPS = (3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4350)


def run_py(args: list[str]) -> str:
    out = subprocess.run(
        [str(PY), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(args)}\n{out.stderr[-2000:]}")
    return out.stdout


def score_step(step: int) -> float:
    ckpt = RUN / "checkpoints" / f"ckpt_step_{step}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing checkpoint: {ckpt}")
    bundle = RUN / f"bundle_diag_step{step}"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])
    out = DIAG / f"diag_step{step}.json"
    run_py([
        "-m", "autotrain.eval_lm", "score",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--out", str(out), "--seq-len", "128", "--batch-size", "16",
    ])
    rep = json.loads(out.read_text(encoding="utf-8"))["report"]
    if rep["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError(f"hash drift at step {step}: {rep['eval_lm_hash']}")
    return rep["scores"]["perplexity"]


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    # Training-internal val loss (from log), for side-by-side comparison.
    import re
    vpat = re.compile(r"step (\d+) \| loss [\d.]+ \| val ([\d.]+)")
    val = {}
    for line in (RUN / "train.log").read_text(encoding="utf-8").splitlines():
        m = vpat.search(line)
        if m:
            val[int(m.group(1))] = float(m.group(2))

    rows = []
    for step in STEPS:
        ppl = score_step(step)
        rows.append({
            "step": step,
            "frozen_holdout_perplexity": ppl,
            "train_internal_val_loss": val.get(step),
        })
        print(f"[step {step}] frozen ppl={ppl:.3f}  "
              f"train val_loss={val.get(step)}")

    ppl_best_step = min(rows, key=lambda r: r["frozen_holdout_perplexity"])["step"]
    post = [r for r in rows if r["step"] > 3750]
    ppl_3750 = min(r["frozen_holdout_perplexity"] for r in rows if r["step"] <= 3800)
    ppl_final = rows[-1]["frozen_holdout_perplexity"]

    summary = {
        "question": "Was the step-4350 early stop premature on the frozen ruler?",
        "eval_lm_hash": EXPECTED_HASH,
        "early_stop": {"triggered_at_step": 4350,
                       "train_val_best_step": 3750,
                       "patience_evaluations": 12,
                       "min_delta": 1e-5},
        "frozen_holdout_best_step": ppl_best_step,
        "frozen_ppl_at_3800": ppl_3750,
        "frozen_ppl_at_final": ppl_final,
        "frozen_ppl_drop_after_train_val_best": ppl_3750 - ppl_final,
        "verdict": ("PREMATURE: frozen ppl keeps dropping after the "
                    "train-val best, so tail-val early stop fired too early"
                    if ppl_3750 - ppl_final > 0.5 else "consistent"),
        "curve": rows,
    }
    (DIAG / "earlystop_diagnosis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"},
                     ensure_ascii=False, indent=2))
    print(f"[written] -> {DIAG / 'earlystop_diagnosis.json'}")


if __name__ == "__main__":
    main()
