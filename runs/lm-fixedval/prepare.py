"""Prepare / verify the clean-retrain run for the val-sampling fix.

The fix in ``autotrain/trainer/{data,runner}.py`` is measurement-only: val is
computed under ``no_grad`` and never touches weights, optimizer, scheduler or
the data cursor. Therefore a clean retrain must reproduce lm-scaled3's
training trajectory bit-for-bit, changing ONLY the early-stop signal.

To make that guarantee auditable, this run keeps the two training-side inputs
byte-identical to the source run and records what is (not) allowed to differ.

Usage:
    python runs/lm-fixedval/prepare.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs/lm-scaled3"
DST = ROOT / "runs/lm-fixedval"
EXPECTED_HASH = "2b70b6f6793432ab"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    (DST / "reports").mkdir(parents=True, exist_ok=True)
    (DST / "checkpoints").mkdir(parents=True, exist_ok=True)

    digests: dict[str, str] = {}
    for name in ("config.yaml", "train_tokens.bin"):
        s, d = SRC / name, DST / name
        if not d.is_file():
            shutil.copy2(s, d)
        a, b = sha256(s), sha256(d)
        if a != b:
            raise RuntimeError(f"{name} sha256 mismatch: src={a} dst={b}")
        digests[name] = a
        print(f"[identical] {name} sha256={a} size={d.stat().st_size}")

    prov = {
        "purpose": "clean retrain validating the tail-val sampling / early-stop fix",
        "source_run": "runs/lm-scaled3",
        "config_yaml": "byte-identical to source (same config_hash, seed, model, budget=4800)",
        "train_tokens_bin": "byte-identical to source (same frozen eval corpus)",
        "frozen_eval_lm_hash": EXPECTED_HASH,
        "sha256": digests,
        "what_changed": [
            "autotrain/trainer/data.py: get_val_batch windows become disjoint & evenly spread",
            "autotrain/trainer/runner.py: _validate samples VAL_WINDOWS=16 disjoint windows",
            "tests/test_data.py + tests/test_runner.py: 4 new regression tests",
        ],
        "what_MUST_NOT_change": [
            "training trajectory: weights / optimizer / scheduler / data cursor",
            "model, optimizer, seed, LR schedule, loss spike detector",
            "min_delta and early_stop_patience (single-variable discipline)",
            "the FROZEN eval_lm_hash = 2b70b6f6793432ab",
        ],
        "expected": (
            "steps 3600/4000 (and 4350 if present) reproduce ext3 perplexity "
            "bit-for-bit; the full 4800-step budget becomes reachable and the "
            "frozen-ruler perplexity keeps falling past 4350"
        ),
        "note": (
            "Repair #3 in EARLY_STOP_REVIEW.md (relative min_delta / smoothing) "
            "is intentionally NOT applied this round, so any behavioural change "
            "is attributable to the sampling fix alone. Re-evaluate only if the "
            "clean retrain still stops prematurely."
        ),
    }
    (DST / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[write] provenance.json")


if __name__ == "__main__":
    main()
