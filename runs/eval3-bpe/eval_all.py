"""Evaluate BPE candidates on the SAME frozen eval-3 questions as char.

Usage:
  python runs/eval3-bpe/eval_all.py char   # reseal char step1350 report with tokenizer fields
  python runs/eval3-bpe/eval_all.py bpe    # evaluate all exported BPE bundles
  python runs/eval3-bpe/eval_all.py compare  # cross-tokenizer A/B on best BPE
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.eval3 import compare_tokenizers  # noqa: E402
from autotrain.eval3.runner import Eval3Runner  # noqa: E402

BASE = ROOT / "runs/eval3-bpe"
CHAR_BASE = ROOT / "runs/eval3-clean"
REPORTS = BASE / "reports"
EVAL3 = ROOT / "evals3"


def evaluate(bundle: Path, tokenizer: Path, out: Path, device: str = "cpu") -> dict:
    tmpl = tokenizer.parent / "chat_template.json"
    runner = Eval3Runner(str(bundle), str(tokenizer), str(tmpl),
                         eval_dir=str(EVAL3), device=device)
    report = runner.run()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"EVAL {out.name}: status={report['status']} step={report['bundle_step']} "
          f"tok={report['tokenizer_type']} "
          f"composite={report['metrics']['composite']:.4f} "
          f"exact_match={report['metrics']['exact_match']:.4f} "
          f"char_f1={report['metrics']['char_f1']:.4f} "
          f"control={report['negative_control']['composite']:.4f} "
          f"empty={report['empty_reply_rate']:.4f}")
    return report


def run_char() -> None:
    bundle = CHAR_BASE / "bundle_step_1350"
    tok = CHAR_BASE / "sft_corpus/tokenizer.json"
    evaluate(bundle, tok, REPORTS / "eval3_char_step1350_resealed.json")


def run_bpe() -> None:
    tok = BASE / "sft_corpus/tokenizer.json"
    bundles = sorted(BASE.glob("bundle_step_*"),
                     key=lambda p: int(p.name.split("_")[-1]))
    if not bundles:
        raise SystemExit("no BPE bundle exported yet")
    for b in bundles:
        evaluate(b, tok, REPORTS / f"eval3_bpe_{b.name}.json")


def run_compare() -> None:
    char = json.loads((REPORTS / "eval3_char_step1350_resealed.json").read_text())
    candidates = sorted(REPORTS.glob("eval3_bpe_*.json"),
                        key=lambda p: json.loads(p.read_text())["metrics"]["composite"],
                        reverse=True)
    if not candidates:
        raise SystemExit("no BPE report")
    best = json.loads(candidates[0].read_text())
    result = compare_tokenizers(char, best)
    (REPORTS / "compare_tok_char_vs_bpe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("COMPARE=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "bpe"
    {"char": run_char, "bpe": run_bpe, "compare": run_compare}[which]()


if __name__ == "__main__":
    main()
