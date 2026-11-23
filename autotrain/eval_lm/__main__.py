"""CLI: python -m autotrain.eval_lm {score,compare,generate}

    python -m autotrain.eval_lm score --bundle B --corpus C --out report.json
    python -m autotrain.eval_lm compare --baseline a.json --candidate b.json
    python -m autotrain.eval_lm generate --bundle B --corpus C --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import compare_reports, generate_continuation, score_bundle


def _read_report(path: str) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc.get("report", doc)


def cmd_score(args: argparse.Namespace) -> None:
    report = score_bundle(
        args.bundle, args.corpus, seq_len=args.seq_len,
        batch_size=args.batch_size, device=args.device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report}, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out.resolve()),
        "eval_lm_hash": report["eval_lm_hash"],
        "step": report["bundle"]["step"],
        **report["scores"],
    }, ensure_ascii=False, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    result = compare_reports(
        _read_report(args.baseline), _read_report(args.candidate),
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    result = generate_continuation(
        args.bundle, args.corpus, args.prompt,
        max_new_tokens=args.max_new_tokens, device=args.device,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="General-purpose LM evaluation")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("score", help="score a bundle on the frozen LM holdout")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--corpus", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--device", default=None)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("compare", help="strict same-hash perplexity comparison")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.add_argument("--threshold", type=float, default=0.0)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("generate", help="greedy continuation (diagnostic)")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--corpus", required=True)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=48)
    sp.add_argument("--device", default=None)
    sp.add_argument(
        "--tokenizer", default=None,
        help="optional model-side tokenizer for generation only; scoring still "
             "uses the frozen corpus tokenizer and eval_lm_hash",
    )
    sp.set_defaults(func=cmd_generate)
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
