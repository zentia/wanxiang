"""100M SFT: seed from M2 step-600, answer-side loss; re-running resumes.

Usage:
    python runs/eval3-large/train_sft.py [target_step]

Config hash MUST match config_m2.yaml (same name/model/data/recover/eval;
only train.max_steps differs, excluded from hash). SFT tokens + labels are
reused from eval3-clean (holdout excluded, answer-side labels present).
"""

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.config import load_config  # noqa: E402
from autotrain.store.experiment_store import ExperimentStore  # noqa: E402
from autotrain.trainer.checkpoint import load_checkpoint  # noqa: E402
from autotrain.trainer.runner import TrainerRunner  # noqa: E402

BASE = ROOT / "runs/eval3-large"
CFG = BASE / "config_sft.yaml"
SFT_TOKENS = ROOT / "runs/eval3-clean/sft_corpus/sft_tokens.bin"
SFT_LABELS = ROOT / "runs/eval3-clean/sft_corpus/sft_labels.bin"
M2_CKPT = BASE / "m2_train/checkpoints/ckpt_step_600.pt"
SEED_CKPT = 600


def main() -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        raw["train"]["max_steps"] = int(sys.argv[1])
    rt = BASE / "config_sft_runtime.yaml"
    rt.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg = load_config(rt)

    ckpt_dir = BASE / "sft_train/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / f"ckpt_step_{SEED_CKPT}.pt"
    if not dst.exists():
        if not M2_CKPT.exists():
            raise SystemExit(f"missing M2 seed checkpoint: {M2_CKPT}")
        shutil.copy2(M2_CKPT, dst)
    probe = TrainerRunner(cfg, str(SFT_TOKENS), str(BASE / "sft_probe"),
                          device="cpu", label_file=str(SFT_LABELS))
    ck = load_checkpoint(dst, cfg, probe.model)
    print(f"SEEDED_FROM_M2 step={ck.step} hash={ck.config_hash[:12]}", flush=True)

    store = ExperimentStore(str(BASE / "experiments_sft.db"))
    r = TrainerRunner(cfg, str(SFT_TOKENS), str(BASE / "sft_train"),
                      store=store, device="mps", label_file=str(SFT_LABELS))
    resumed = r.resume()
    print(f"SFT_RESUME={resumed} step={r.step}", flush=True)
    summary = r.train()
    print("SFT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
