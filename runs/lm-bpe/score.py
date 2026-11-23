"""Score the same-hash BPE model on the FROZEN char holdout.

Scoring deliberately does NOT re-tokenise the holdout: eval_lm always loads
runs/corpus/lm/tokenizer.json, so the report keeps eval_lm_hash=2b70b6f6793432ab
and is strictly comparable with the char step1200 model.

Generation is only a diagnostic; there we pass runs/lm-bpe/tokenizer.json so
subword outputs can be decoded correctly.

Usage:
    python runs/lm-bpe/score.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "runs/corpus/lm"          # FROZEN — do not change
CONFIG = ROOT / "runs/lm-bpe/config.yaml"
RUN = ROOT / "runs/lm-bpe"
TOKENIZER = RUN / "tokenizer.json"
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
        raise RuntimeError("no checkpoints in runs/lm-bpe")
    final_ckpt = ckpts[-1]
    bundle = RUN / "bundle_bpe"
    run_py([
        "-m", "autotrain.export", "export",
        "--checkpoint", str(final_ckpt), "--config", str(CONFIG),
        "--out", str(bundle),
    ])

    out = REPORTS / "lm_bpe.json"
    run_py([
        "-m", "autotrain.eval_lm", "score",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--out", str(out), "--seq-len", "128", "--batch-size", "16",
    ])
    cand = json.loads(out.read_text(encoding="utf-8"))["report"]
    if cand["eval_lm_hash"] != EXPECTED_HASH:
        raise RuntimeError(
            f"eval_lm_hash drifted: {cand['eval_lm_hash']} != {EXPECTED_HASH}; "
            "BPE must be scored on the frozen char holdout without retokenisation"
        )

    ref = json.loads(REF_REPORT.read_text(encoding="utf-8"))["report"]
    cmp = run_py([
        "-m", "autotrain.eval_lm", "compare",
        "--baseline", str(REF_REPORT), "--candidate", str(out),
    ])
    (REPORTS / "compare_step1200_char_vs_bpe.json").write_text(cmp + "\n", encoding="utf-8")
    print(f"[ref char step1200] ppl={ref['scores']['perplexity']:.2f} ce={ref['scores']['cross_entropy']:.4f}")
    print(f"[candidate BPE {cand['bundle']['step']}] ppl={cand['scores']['perplexity']:.2f} ce={cand['scores']['cross_entropy']:.4f}")
    print("[compare]\n" + cmp)

    gen = run_py([
        "-m", "autotrain.eval_lm", "generate",
        "--bundle", str(bundle), "--corpus", str(CORPUS),
        "--tokenizer", str(TOKENIZER),
        "--prompt", "自然语言处理是", "--max-new-tokens", "48",
    ])
    (REPORTS / "continuation_bpe.json").write_text(gen + "\n", encoding="utf-8")
    print("[generate]\n" + gen)


if __name__ == "__main__":
    main()
