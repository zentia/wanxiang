"""CLI: python -m autotrain.eval3 {freeze,verify,run,compare}

    python -m autotrain.eval3 freeze --source runs/m2/sources/idiom.json \
        --tokenizer runs/sft/corpus/tokenizer.json \
        --chat-template runs/sft/corpus/chat_template.json --out evals3
    python -m autotrain.eval3 verify
    python -m autotrain.eval3 run --bundle <dir> --out <report.json>
    python -m autotrain.eval3 compare <before.json> <after.json>
    python -m autotrain.eval3 compare-tok <char.json> <bpe.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..inference import ChatTemplate, load_tokenizer
from .corpus import EVAL3_DIR, freeze, verify_corpus
from .runner import Eval3Runner, compare_reports, compare_tokenizers

DEFAULT_SOURCE = "runs/m2/sources/idiom.json"
DEFAULT_TOKENIZER = "runs/sft/corpus/tokenizer.json"
DEFAULT_TEMPLATE = "runs/sft/corpus/chat_template.json"


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def cmd_freeze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if (out / "manifest.json").is_file() and not args.force:
        raise SystemExit(
            f"{out} is already frozen; pass --force only if you intend to "
            "invalidate every existing eval-3 report."
        )
    tok = load_tokenizer(args.tokenizer)
    tmpl = ChatTemplate.from_file(args.chat_template)
    doc = freeze(args.source, tok, tmpl, out)
    _print({
        "status": "FROZEN",
        "eval_dir": str(out.resolve()),
        "corpus_manifest_hash": doc["corpus_manifest_hash"],
        "qa_count": doc["qa_count"],
        "holdout_seed": doc["holdout_seed"],
        "holdout_count": doc["holdout_count"],
        "files": sorted(doc["files"]),
    })


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify_corpus(args.eval_dir)
    _print({
        "status": "OK",
        "eval_dir": str(Path(args.eval_dir or EVAL3_DIR).resolve()),
        "corpus_manifest_hash": doc["corpus_manifest_hash"],
        "qa_count": doc["qa_count"],
        "holdout_count": doc["holdout_count"],
        "frozen": doc["frozen"],
        "files": sorted(doc["files"]),
    })


def cmd_run(args: argparse.Namespace) -> None:
    runner = Eval3Runner(
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
        "exact_match": round(report["metrics"]["exact_match"], 4),
        "control_composite": round(report["negative_control"]["composite"], 4),
        "empty_reply_rate": round(report["empty_reply_rate"], 4),
        "out": args.out,
    })


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    _print(compare_reports(before, after))


def cmd_compare_tok(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    _print(compare_tokenizers(before, after))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wanxiang eval-3 (word-holdout QA instrument)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("freeze", help="one-time freeze of evals3/")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    sp.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    sp.add_argument("--out", default=str(EVAL3_DIR))
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

    sp = sub.add_parser("compare", help="A/B two eval-3 reports (strict)")
    sp.add_argument("before")
    sp.add_argument("after")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("compare-tok", help="A/B across a tokenizer change")
    sp.add_argument("before")
    sp.add_argument("after")
    sp.set_defaults(func=cmd_compare_tok)
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
