"""Score the ext3 + long-budget model on the FROZEN eval_lm instrument.

The 4800-step budget hit early-stop (patience 12 evaluations) at step 4350,
so there is no step4800 checkpoint. We DISCOVER the final step from disk and
score a curve of checkpoints on the SAME frozen ruler:
  * step 3600  — mid-budget snapshot (user-requested lower bound)
  * step 4000  — intermediate
  * final step — the checkpoint the run actually converged at (4350)

All are compared against runs/lm-scaled2 step2400 (ppl 56.62), so the only
changes are more disjoint train documents and a longer step budget. The eval
corpus, holdout, tokenizer and hash stay byte-identical (2b70b6f6793432ab).

A step3600 -> final comparison isolates whether gains continue to accrue in
the last stretch, i.e. whether the curve is still dropping or has plateaued.

NOTE: the training-internal val_loss is NOT comparable across these runs —
val_fraction slices the tail of each (different) train stream. Only the frozen
holdout perplexity is a valid cross-run judge.

Usage:
    python runs/lm-scaled3/score.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"          # FROZEN — do not change
CONFIG = ROOT / "runs/lm-scaled3/config.yaml"
RUN = ROOT / "runs/lm-scaled3"
REF_REPORT = ROOT / "runs/lm-scaled2/reports/lm_scaled2.json"
EXPECTED_HASH = "2b70b6f6793432ab"
MILESTONES = (3600, 4000)   # explicit mid-budget snapshots
REPORTS = RUN / "reports"
PY = ROOT / ".venv/bin/python"


def _discover_steps() -> list[int]:
    """Sorted numeric checkpoint steps; final = actual early-stop step."""
    ckpts = sorted(
        (RUN / "checkpoints").glob("ckpt_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise RuntimeError("no checkpoints in runs/lm-scaled3")
    available = [int(p.stem.rsplit("_", 1)[-1]) for p in ckpts]
    final = available[-1]
    steps = sorted({s for s in (*MILESTONES, final) if s in available})
    if final not in steps:  # always include the real convergence point
        steps.append(final)
        steps.sort()
    return steps


def run_py(args: list[str]) -> str:
    out = subprocess.run(
        [str(PY), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(args)}\n{out.stderr[-2000:]}")
    return out.stdout


def _score_step(step: int) -> dict:
    ckpt = RUN / "checkpoints" / f"ckpt_step_{step}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing checkpoint for step {step}: {ckpt}")
    bundle = RUN / f"bundle_scaled3_step{step}"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])

    out = REPORTS / f"lm_scaled3_step{step}.json"
    run_py([
        "-m", "autotrain.eval_lm", "score",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--out", str(out), "--seq-len", "128", "--batch-size", "16",
    ])
    cand = json.loads(out.read_text(encoding="utf-8"))["report"]
    if cand["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError(
            f"eval_lm_hash drifted at step {step}: {cand['eval_lm_hash']} != "
            f"{EXPECTED_HASH}; A/B invalid — aborting"
        )
    return cand


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    ref = json.loads(REF_REPORT.read_text(encoding="utf-8"))["report"]
    if ref["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError("reference step2400 report is not on the frozen hash")

    steps = _discover_steps()
    final_step = steps[-1]
    print(f"[discover] scoring steps={steps} (final/early-stop={final_step}; "
          f"4800 budget never reached)")
    print(f"[ref step2400] ppl={ref['scores']['perplexity']:.2f} "
          f"ce={ref['scores']['cross_entropy']:.4f}")

    curve = []
    for step in steps:
        cand = _score_step(step)
        out = REPORTS / f"lm_scaled3_step{step}.json"
        cmp = run_py([
            "-m", "autotrain.eval_lm", "compare",
            "--baseline", str(REF_REPORT), "--candidate", str(out),
        ])
        cmp_doc = json.loads(cmp)
        (REPORTS / f"compare_step2400_vs_step{step}.json").write_text(
            cmp + "\n", encoding="utf-8"
        )
        ppl = cand["scores"]["perplexity"]
        curve.append({
            "step": step,
            "perplexity": ppl,
            "cross_entropy": cand["scores"]["cross_entropy"],
            "verdict_vs_step2400": cmp_doc["verdict"],
        })
        print(f"[candidate step{step}] ppl={ppl:.2f} "
              f"ce={cand['scores']['cross_entropy']:.4f} -> {cmp_doc['verdict']}")

    # Plateau check: does the last stretch (3600 -> final) still improve?
    plateau_cmp = None
    if len(steps) >= 2 and steps[0] < final_step:
        mid_out = REPORTS / f"lm_scaled3_step{steps[0]}.json"
        final_out = REPORTS / f"lm_scaled3_step{final_step}.json"
        pc = run_py([
            "-m", "autotrain.eval_lm", "compare",
            "--baseline", str(mid_out), "--candidate", str(final_out),
        ])
        (REPORTS / f"compare_step{steps[0]}_vs_step{final_step}.json").write_text(
            pc + "\n", encoding="utf-8"
        )
        plateau_cmp = json.loads(pc)
        print(f"[plateau step{steps[0]} -> step{final_step}]\n" + pc)

    # Generation diagnostic on the final converged snapshot only.
    bundle = RUN / f"bundle_scaled3_step{final_step}"
    gen = run_py([
        "-m", "autotrain.eval_lm", "generate",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--prompt", "自然语言处理是", "--max-new-tokens", "48",
    ])
    (REPORTS / f"continuation_step{final_step}.json").write_text(gen + "\n", encoding="utf-8")
    print(f"[generate step{final_step}]\n" + gen)

    summary = {
        "experiment": "wanxiang-lm-scaled3",
        "eval_lm_hash": EXPECTED_HASH,
        "budget_max_steps": 4800,
        "actual_final_step": final_step,
        "early_stopped": final_step < 4800,
        "reference": {"step": 2400, "perplexity": ref["scores"]["perplexity"]},
        "curve": curve,
        "plateau_step3600_to_final": plateau_cmp,
        "note": "training-internal val_loss is NOT cross-run comparable "
                "(val_fraction slices each train stream's tail); only frozen "
                "holdout perplexity is a valid judge",
    }
    (REPORTS / "lm_scaled3_curve.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[summary] -> {REPORTS / 'lm_scaled3_curve.json'}")


if __name__ == "__main__":
    main()
