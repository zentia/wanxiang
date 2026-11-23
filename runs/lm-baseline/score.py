"""Score the LM baseline (step0 vs trained) on the frozen wiki holdout.

Usage (after training finishes and checkpoints are exported to bundles):
    python runs/lm-baseline/score.py

Exports the final checkpoint to a bundle if needed, scores both bundles with
eval-lm, writes a strict comparison report, and emits greedy continuations
as a qualitative diagnostic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"
CONFIG = ROOT / "runs/lm-baseline/config.yaml"
RUN = ROOT / "runs/lm-baseline"
REPORTS = RUN / "reports"
PY = ROOT / ".venv/bin/python"


def run_py(args: list[str]) -> str:
    out = subprocess.run(
        [str(PY), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(args)}\n{out.stderr[-2000:]}")
    return out.stdout


def latest_checkpoint() -> Path | None:
    ckpts = sorted((RUN / "checkpoints").glob("ckpt_step_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Export final checkpoint to a bundle if not already present.
    ckpt = latest_checkpoint()
    trained_bundle = RUN / "bundle_trained"
    if ckpt is not None and not (trained_bundle / "model.safetensors").is_file():
        run_py([
            "-m", "autotrain.export", "export",
            "--checkpoint", str(ckpt), "--config", str(CONFIG),
            "--out", str(trained_bundle),
        ])

    bundles = {"step0": RUN / "bundle_step0"}
    if (trained_bundle / "model.safetensors").is_file():
        bundles["trained"] = trained_bundle

    reports = {}
    for name, bundle in bundles.items():
        out = REPORTS / f"lm_{name}.json"
        stdout = run_py([
            "-m", "autotrain.eval_lm", "score",
            "--bundle", str(bundle), "--corpus", str(CORPUS),
            "--out", str(out), "--seq-len", "128", "--batch-size", "16",
        ])
        print(f"[{name}] {stdout.strip()}")
        reports[name] = json.loads(out.read_text(encoding="utf-8"))["report"]

    if "trained" in reports:
        cmp = run_py([
            "-m", "autotrain.eval_lm", "compare",
            "--baseline", str(REPORTS / "lm_step0.json"),
            "--candidate", str(REPORTS / "lm_trained.json"),
        ])
        (REPORTS / "compare_step0_vs_trained.json").write_text(
            cmp + "\n", encoding="utf-8"
        )
        print("[compare]\n" + cmp)

        # Qualitative continuation from the trained model.
        gen = run_py([
            "-m", "autotrain.eval_lm", "generate",
            "--bundle", str(trained_bundle), "--corpus", str(CORPUS),
            "--prompt", "自然语言处理是", "--max-new-tokens", "48",
        ])
        (REPORTS / "continuation.json").write_text(gen + "\n", encoding="utf-8")
        print("[generate]\n" + gen)


if __name__ == "__main__":
    main()
