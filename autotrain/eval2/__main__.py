"""CLI: python -m autotrain.eval2 {freeze,verify,run,compare}

    python -m autotrain.eval2 freeze        # one-time freeze of evals2/
    python -m autotrain.eval2 verify        # byte-check the frozen set
    python -m autotrain.eval2 run --bundle runs/sft/bundle --out runs/eval2/report_sft.json
    python -m autotrain.eval2 compare runs/eval2/report_m2.json runs/eval2/report_sft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..inference import CharTokenizer, ChatTemplate
from .corpus import EVAL2_DIR, freeze, verify_corpus
from .runner import Eval2Runner, compare_reports

DEFAULT_SFT = "runs/sft/corpus/sft_tokens.bin"
DEFAULT_TOKENIZER = "runs/sft/corpus/tokenizer.json"
DEFAULT_TEMPLATE = "runs/sft/corpus/chat_template.json"


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def cmd_freeze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if (out / "manifest.json").is_file() and not args.force:
        raise SystemExit(
            f"{out} is already frozen; pass --force only if you intend to "
            "invalidate every existing eval-2 report."
        )
    tok = CharTokenizer.from_file(args.tokenizer)
    tmpl = ChatTemplate.from_file(args.chat_template)
    doc = freeze(args.sft_tokens, tok, tmpl, out)
    _print({
        "status": "FROZEN",
        "eval_dir": str(out.resolve()),
        "corpus_manifest_hash": doc["corpus_manifest_hash"],
        "qa_count": doc["qa_count"],
        "min_source_offset": doc["min_source_offset"],
        "files": sorted(doc["files"]),
    })


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify_corpus(args.eval_dir)
    _print({
        "status": "OK",
        "eval_dir": str(Path(args.eval_dir or EVAL2_DIR).resolve()),
        "corpus_manifest_hash": doc["corpus_manifest_hash"],
        "qa_count": doc["qa_count"],
        "frozen": doc["frozen"],
        "files": sorted(doc["files"]),
    })


def cmd_run(args: argparse.Namespace) -> None:
    runner = Eval2Runner(
        args.bundle, args.tokenizer, args.chat_template,
        eval_dir=args.eval_dir, device=args.device, threads=args.threads,
    )
    report = runner.run()
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                sort_keys=True) + "\n", encoding="utf-8")
    _print({
        "status": report["status"],
        "step": report["bundle_step"],
        "evalspec_hash": report["evalspec_hash"][:16],
        "composite": round(report["metrics"]["composite"], 4),
        "control_composite": round(report["negative_control"]["composite"], 4),
        "empty_reply_rate": round(report["empty_reply_rate"], 4),
        "out": args.out,
    })


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    _print(compare_reports(before, after))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wanxiang eval-2 (frozen text QA instrument)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("freeze", help="one-time freeze of evals2/")
    sp.add_argument("--sft-tokens", default=DEFAULT_SFT)
    sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    sp.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    sp.add_argument("--out", default=str(EVAL2_DIR))
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_freeze)

    sp = sub.add_parser("verify", help="verify the frozen corpus")
    sp.add_argument("--eval-dir", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("run", help="evaluate a bundle")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    sp.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    sp.add_argument("--eval-dir", default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--threads", type=int, default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("compare", help="A/B two eval-2 reports")
    sp.add_argument("before")
    sp.add_argument("after")
    sp.set_defaults(func=cmd_compare)
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
