"""Score the clean retrain on the FROZEN eval_lm ruler and verify the prefix.

The val-sampling fix is measurement-only (no_grad; no weight/cursor/optimizer
updates), so the early steps MUST reproduce lm-scaled3 bit-for-bit. The new
evidence is the tail past 4350: once the PREMATURE early stop is removed, does
frozen-holdout perplexity keep falling with the full 4800-step budget?

Checks performed:
  * eval_lm_hash stays 2b70b6f6793432ab on every scored checkpoint;
  * steps 3600/4000/4350 (where available) match ext3 perplexity within 1e-6;
  * plateau comparison from the last pre-4350 milestone to the final step;
  * writes reports/lm_fixedval_curve.json and exits 2 if the prefix drifted.

Usage:
    python runs/lm-fixedval/score.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"                    # FROZEN — do not change
CONFIG = ROOT / "runs/lm-fixedval/config.yaml"
RUN = ROOT / "runs/lm-fixedval"
SRC_RUN = ROOT / "runs/lm-scaled3"
REPORTS = RUN / "reports"
REF_REPORT = ROOT / "runs/lm-scaled2/reports/lm_scaled2.json"
EXPECTED_HASH = "2b70b6f6793432ab"
MILESTONES = (3600, 4000, 4350, 4800)
PREFIX_STEPS = (3600, 4000, 4350)   # must match ext3 bit-for-bit where present
PY = ROOT / ".venv/bin/python"


def run_py(args: list[str]) -> str:
    out = subprocess.run(
        [str(PY), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(args)}\n{out.stderr[-2000:]}")
    return out.stdout


def discover_steps() -> tuple[list[int], int]:
    ckpts = sorted(
        (RUN / "checkpoints").glob("ckpt_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise RuntimeError("no checkpoints in runs/lm-fixedval")
    available = [int(p.stem.rsplit("_", 1)[-1]) for p in ckpts]
    final = available[-1]
    steps = sorted({s for s in (*MILESTONES, final) if s in available})
    if final not in steps:
        steps.append(final)
        steps.sort()
    return steps, final


def score_step(step: int) -> dict:
    ckpt = RUN / "checkpoints" / f"ckpt_step_{step}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing checkpoint for step {step}: {ckpt}")
    bundle = RUN / f"bundle_fixedval_step{step}"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])
    out = REPORTS / f"lm_fixedval_step{step}.json"
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

    steps, final = discover_steps()
    print(f"[discover] steps={steps} final={final} "
          f"reached_full_budget={final >= 4800}")
    print(f"[ref step2400] ppl={ref['scores']['perplexity']:.2f}")

    curve = []
    for step in steps:
        cand = score_step(step)
        out = REPORTS / f"lm_fixedval_step{step}.json"
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
        print(f"[step{step}] ppl={ppl:.4f} "
              f"ce={cand['scores']['cross_entropy']:.4f} -> {cmp_doc['verdict']}")

    # Prefix identity: the fix must NOT alter the training trajectory.
    prefix = []
    for step in PREFIX_STEPS:
        src = SRC_RUN / "reports" / f"lm_scaled3_step{step}.json"
        cur = REPORTS / f"lm_fixedval_step{step}.json"
        if not (src.is_file() and cur.is_file()):
            continue
        a = json.loads(src.read_text(encoding="utf-8"))["report"]["scores"]["perplexity"]
        b = json.loads(cur.read_text(encoding="utf-8"))["report"]["scores"]["perplexity"]
        prefix.append({
            "step": step, "ext3_ppl": a, "fixedval_ppl": b,
            "delta": b - a, "identical": abs(a - b) < 1e-6,
        })
    prefix_ok = bool(prefix) and all(p["identical"] for p in prefix)
    print(f"[prefix] reproduces_ext3={prefix_ok} ({len(prefix)} checkpoints)")
    for p in prefix:
        print(f"  step{p['step']}: ext3={p['ext3_ppl']:.4f} "
              f"fixed={p['fixedval_ppl']:.4f} delta={p['delta']:+.2e}")

    # Plateau: last pre-tail milestone -> final step.
    plateau = None
    if len(steps) >= 2:
        base = next((s for s in (4350, 4000, 3600) if s in steps), steps[0])
        base_out = REPORTS / f"lm_fixedval_step{base}.json"
        final_out = REPORTS / f"lm_fixedval_step{final}.json"
        if base < final and base_out.is_file() and final_out.is_file():
            pc = run_py([
                "-m", "autotrain.eval_lm", "compare",
                "--baseline", str(base_out), "--candidate", str(final_out),
            ])
            (REPORTS / f"compare_step{base}_vs_step{final}.json").write_text(
                pc + "\n", encoding="utf-8"
            )
            plateau = json.loads(pc)
            print(f"[plateau step{base} -> step{final}]\n{pc}")

    summary = {
        "experiment": "wanxiang-lm-fixedval",
        "eval_lm_hash": EXPECTED_HASH,
        "budget_max_steps": 4800,
        "actual_final_step": final,
        "reached_full_budget": final >= 4800,
        "reference": {"step": 2400, "perplexity": ref["scores"]["perplexity"]},
        "curve": curve,
        "prefix_identity_vs_ext3": prefix_ok,
        "prefix_detail": prefix,
        "plateau": plateau,
        "note": (
            "val-sampling fix is measurement-only; scored prefix must reproduce "
            "ext3 bit-for-bit. min_delta/patience/model/seed unchanged "
            "(single-variable discipline). Only frozen holdout perplexity is a "
            "valid cross-run judge; training-internal val_loss is not."
        ),
    }
    (REPORTS / "lm_fixedval_curve.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[summary] -> {REPORTS / 'lm_fixedval_curve.json'}")
    if not prefix_ok:
        print("[WARN] prefix perplexity drifted from ext3 — training "
              "trajectory changed; investigate before trusting the tail")
        sys.exit(2)


if __name__ == "__main__":
    main()
