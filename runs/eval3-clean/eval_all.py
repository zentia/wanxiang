"""Export candidate bundles and evaluate both old and new on eval-3.

Produces reports under runs/eval3-clean/reports/:
  eval3_old_step1700.json   -- the old (leaked-eval2) model on the honest尺子
  eval3_new_step_N.json     -- the clean-retrained model
Then compares old -> new. All use the frozen evals3/ corpus.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.eval3 import compare_reports  # noqa: E402
from autotrain.eval3.runner import Eval3Runner  # noqa: E402

BASE = ROOT / "runs/eval3-clean"
REPORTS = BASE / "reports"
TOK = str(BASE / "sft_corpus/tokenizer.json")
TMPL = str(BASE / "sft_corpus/chat_template.json")
OLD_BUNDLE = ROOT / "runs/sft-long/bundle_1700"


def evaluate(bundle: str | Path, out_name: str, device: str = "cpu") -> dict:
    r = Eval3Runner(str(bundle), TOK, TMPL, device=device)
    report = r.run()
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / out_name
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"EVAL3 {out_name}: status={report['status']} "
          f"composite={report['metrics']['composite']:.4f} "
          f"exact_match={report['metrics']['exact_match']:.4f} "
          f"control={report['negative_control']['composite']:.4f}")
    return report


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "old"
    if which == "old":
        evaluate(OLD_BUNDLE, "eval3_old_step1700.json")
    elif which == "new":
        # find the latest exported clean bundle
        bundles = sorted(BASE.glob("bundle_step_*"),
                         key=lambda p: int(p.name.split("_")[-1]))
        if not bundles:
            raise SystemExit("no clean bundle exported yet")
        latest = bundles[-1]
        evaluate(latest, f"eval3_new_{latest.name}.json")
    elif which == "compare":
        old = json.loads((REPORTS / "eval3_old_step1700.json").read_text())
        news = sorted(REPORTS.glob("eval3_new_*.json"))
        if not news:
            raise SystemExit("no new report")
        new = json.loads(news[-1].read_text())
        print(json.dumps(compare_reports(old, new), ensure_ascii=False,
                         indent=2, sort_keys=True))
    else:
        raise SystemExit("usage: eval_all.py {old|new|compare}")


if __name__ == "__main__":
    main()
