"""Clean SFT v2: resume M2 step-600 ckpt, answer-side loss, holdout excluded."""

import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.config import load_config  # noqa: E402
from autotrain.store.experiment_store import ExperimentStore  # noqa: E402
from autotrain.trainer.checkpoint import load_checkpoint  # noqa: E402
from autotrain.trainer.model import init_model  # noqa: E402
from autotrain.trainer.runner import TrainerRunner  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

BASE = ROOT / "runs/eval3-clean"
CFG_SRC = ROOT / "autotrain/configs/wanxiang-m2-mac-long.yaml"
SFT_TOKENS = BASE / "sft_corpus/sft_tokens.bin"
SFT_LABELS = BASE / "sft_corpus/sft_labels.bin"
M2_CKPT = BASE / "m2_train/checkpoints/ckpt_step_600.pt"
MAX_STEPS = 1500            # budget knob; early stop guards overfitting
SEED_CKPT = 600


def write_cfg(max_steps: int, out: Path) -> str:
    raw = yaml.safe_load(CFG_SRC.read_text(encoding="utf-8"))
    raw["train"]["max_steps"] = max_steps
    raw["experiment"] = "wanxiang-eval3-clean-m2"  # MUST match M2 config hash
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return str(out)


def main() -> None:
    cfg_path = write_cfg(MAX_STEPS, BASE / "config_sft.yaml")
    cfg = load_config(cfg_path)

    # Seed the SFT run's checkpoint dir with the M2 step-600 checkpoint so
    # TrainerRunner.resume() picks it up. Same config_hash -> legal transfer.
    ckpt_dir = BASE / "sft_train/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / f"ckpt_step_{SEED_CKPT}.pt"
    if not dst.exists():
        shutil.copy2(M2_CKPT, dst)
    # Verify the transferred checkpoint is compatible before training.
    probe = TrainerRunner(cfg, str(SFT_TOKENS), str(BASE / "sft_train_probe"),
                          device="cpu", label_file=str(SFT_LABELS))
    ck = load_checkpoint(dst, cfg, probe.model)
    assert ck.step == SEED_CKPT
    print(f"SEEDED_FROM_M2 step={ck.step} hash={ck.config_hash[:12]}")

    store = ExperimentStore(str(BASE / "experiments_sft.db"))
    r = TrainerRunner(
        cfg, str(SFT_TOKENS), str(BASE / "sft_train"),
        store=store, device="mps", label_file=str(SFT_LABELS),
    )
    assert r.resume(), "SFT must resume from seeded M2 step-600"
    assert r.step == SEED_CKPT
    summary = r.train()
    print("SFT_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
