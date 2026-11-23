"""CLI: python -m autotrain.sft {build,verify,template}

    python -m autotrain.sft build
    python -m autotrain.sft verify
    python -m autotrain.sft template
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..inference.chat import ChatTemplate
from ..inference.tokenizer import CharTokenizer
from .corpus import build, verify

DEFAULT_SOURCE = "runs/m2/sources/idiom.json"
DEFAULT_TOKENIZER = "runs/m2/corpus/tokenizer.json"
DEFAULT_OUT = "runs/sft/corpus"


def cmd_build(args: argparse.Namespace) -> None:
    doc = build(args.source, args.tokenizer, args.out,
                max_entries=args.max_entries)
    print(json.dumps({
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "out": str(Path(args.out).resolve()),
        "entries_used": doc["extraction"]["entries_used"],
        "tokens": doc["files"]["sft_tokens.bin"]["tokens"],
        "chat_template": doc["chat_template"]["template_hash"][:16],
        "vocab_size": doc["tokenizer"]["vocab_size"],
        "files": sorted(doc["files"]),
    }, ensure_ascii=False, indent=2))
    print("\nverify: python -m autotrain.sft verify --corpus " + args.out)


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify(args.corpus)
    print(json.dumps({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "tokens": doc["files"]["sft_tokens.bin"]["tokens"],
        "template": doc["chat_template"]["version"],
        "template_hash": doc["chat_template"]["template_hash"][:16],
    }, ensure_ascii=False, indent=2))


def cmd_template(args: argparse.Namespace) -> None:
    tok = CharTokenizer.from_file(Path(args.corpus) / "tokenizer.json")
    tmpl = ChatTemplate.from_file(Path(args.corpus) / "chat_template.json")
    tmpl.validate_chars(tok)
    sample = tmpl.encode_messages(
        [{"role": "user", "content": "请解释成语《画蛇添足》的意思"}], tok
    )
    print(json.dumps({
        "template_hash": tmpl.template_hash,
        "role_prefix": tmpl.role_prefix,
        "turn_sep": repr(tmpl.turn_sep),
        "generation_prefix": repr(tmpl.generation_prefix),
        "sample_prompt": tmpl.training_text("请解释成语《画蛇添足》的意思",
                                            "比喻做了多余的事。"),
        "sample_prompt_ids_head": sample[:12],
        "chars_validated": True,
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wanxiang SFT prototype corpus")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build", help="build instruction-formatted corpus")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    sp.add_argument("--out", default=DEFAULT_OUT)
    sp.add_argument("--max-entries", type=int, default=None)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("verify", help="verify checksums and seals")
    sp.add_argument("--corpus", default=DEFAULT_OUT)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("template", help="inspect the versioned chat template")
    sp.add_argument("--corpus", default=DEFAULT_OUT)
    sp.set_defaults(func=cmd_template)
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
