"""Clean M2 pretrain from step 0 (holdout words excluded from the stream)."""

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
from autotrain.trainer.runner import TrainerRunner  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

BASE = ROOT / "runs/eval3-clean"
CFG_SRC = ROOT / "autotrain/configs/wanxiang-m2-mac-long.yaml"
M2_TOKENS = BASE / "m2_corpus/corpus_tokens.bin"
MAX_STEPS = 600


def write_cfg(max_steps: int, out: Path) -> str:
    raw = yaml.safe_load(CFG_SRC.read_text(encoding="utf-8"))
    raw["train"]["max_steps"] = max_steps
    raw["experiment"] = "wanxiang-eval3-clean-m2"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return str(out)


def main() -> None:
    # NOTE: experiment name changes config_hash. That is fine for M2 (fresh
    # run), but the SFT stage must use the SAME name+config to resume the
    # step-600 checkpoint -- see train_sft.py, which mirrors this exactly.
    cfg_path = write_cfg(MAX_STEPS, BASE / "config_m2.yaml")
    cfg = load_config(cfg_path)
    store = ExperimentStore(str(BASE / "experiments_m2.db"))
    r = TrainerRunner(
        cfg, str(M2_TOKENS), str(BASE / "m2_train"),
        store=store, device="mps",
    )
    assert not r.resume(), "clean M2 must start from step 0"
    summary = r.train()
    print("M2_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
