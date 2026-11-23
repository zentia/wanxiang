"""100M M2 pretrain from step 0; re-running resumes from latest checkpoint.

Usage:
    python runs/eval3-large/train_m2.py [target_step]

Budget knob: target_step only changes train.max_steps, which is excluded
from config_hash, so every checkpoint stays resume-compatible. Char corpus
is reused from eval3-clean (holdout words already excluded at source).
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.config import load_config  # noqa: E402
from autotrain.store.experiment_store import ExperimentStore  # noqa: E402
from autotrain.trainer.runner import TrainerRunner  # noqa: E402

BASE = ROOT / "runs/eval3-large"
CFG = BASE / "config_m2.yaml"
TOKENS = ROOT / "runs/eval3-clean/m2_corpus/corpus_tokens.bin"


def main() -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        raw["train"]["max_steps"] = int(sys.argv[1])
    rt = BASE / "config_m2_runtime.yaml"
    rt.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    cfg = load_config(rt)

    store = ExperimentStore(str(BASE / "experiments_m2.db"))
    r = TrainerRunner(cfg, str(TOKENS), str(BASE / "m2_train"),
                      store=store, device="mps")
    resumed = r.resume()
    print(f"M2_RESUME={resumed} step={r.step}", flush=True)
    summary = r.train()
    print("M2_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
