"""Export candidate bundles and evaluate on frozen eval3; strict A/B vs
the 10M char clean baseline (composite 0.0676).

Usage:
    python runs/eval3-large/eval_all.py export <step>...
    python runs/eval3-large/eval_all.py eval
    python runs/eval3-large/eval_all.py compare

Same tokenizer (char), same evals3/ frozen set, same device as baseline, so
strict compare_reports() applies -- not the cross-tokenizer protocol.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.eval3 import compare_reports  # noqa: E402
from autotrain.eval3.runner import Eval3Runner  # noqa: E402

BASE = ROOT / "runs/eval3-large"
REPORTS = BASE / "reports"
CFG = BASE / "config_sft.yaml"
TOK = str(ROOT / "runs/eval3-clean/sft_corpus/tokenizer.json")
TMPL = str(ROOT / "runs/eval3-clean/sft_corpus/chat_template.json")
BASELINE = ROOT / "runs/eval3-bpe/reports/eval3_char_step1350_resealed.json"


def do_export(steps: list[str]) -> None:
    for s in steps:
        ckpt = BASE / f"sft_train/checkpoints/ckpt_step_{s}.pt"
        out = BASE / f"bundle_step_{s}"
        subprocess.run([
            sys.executable, "-m", "autotrain.export", "export",
            "--checkpoint", str(ckpt), "--config", str(CFG),
            "--out", str(out), "--force",
        ], cwd=str(ROOT), check=True)
        print(f"exported step {s}")


def do_eval() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    bundles = sorted(BASE.glob("bundle_step_*"),
                     key=lambda p: int(p.name.split("_")[-1]))
    if not bundles:
        raise SystemExit("no bundle exported yet")
    for b in bundles:
        r = Eval3Runner(str(b), TOK, TMPL, device="cpu")
        report = r.run()
        name = f"eval3_large_{b.name}.json"
        (REPORTS / name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"EVAL3 {name}: status={report['status']} "
              f"composite={report['metrics']['composite']:.4f} "
              f"exact_match={report['metrics']['exact_match']:.4f} "
              f"control={report['negative_control']['composite']:.4f}")


def do_compare() -> None:
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    cands = sorted(REPORTS.glob("eval3_large_*.json"))
    if not cands:
        raise SystemExit("no eval3-large report")
    for p in cands:
        cand = json.loads(p.read_text(encoding="utf-8"))
        v = compare_reports(base, cand)
        print(f"\n=== {p.name} ===")
        print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "export":
        do_export(sys.argv[2:])
    elif cmd == "eval":
        do_eval()
    elif cmd == "compare":
        do_compare()
    else:
        raise SystemExit("usage: eval_all.py {export <step>...|eval|compare}")


if __name__ == "__main__":
    main()
