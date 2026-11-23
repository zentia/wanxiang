"""CLI: python -m autotrain.corpus {fetch,build,verify}

    python -m autotrain.corpus fetch                 # download pinned wiki_zh
    python -m autotrain.corpus build                 # fetch(cached) + sealed corpus
    python -m autotrain.corpus verify                # checksum + seal check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import build, verify
from .source import SOURCES, fetch

DEFAULT_SOURCES_DIR = "runs/corpus/sources"
DEFAULT_CORPUS_DIR = "runs/corpus/lm"


def cmd_fetch(args: argparse.Namespace) -> None:
    meta = fetch(args.source, args.sources_dir, force=args.force)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    meta = fetch(args.source, args.sources_dir)  # cached -> no download
    doc = build(
        meta["path"], args.out,
        source_meta=meta,
        vocab_size=args.vocab_size,
        tokenizer_kind=args.tokenizer_kind,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        max_chars=args.max_chars,
    )
    print(json.dumps({
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "out": str(Path(args.out).resolve()),
        "source": doc["source"].get("license"),
        "documents": doc["extraction"]["documents"],
        "train_tokens": doc["files"]["train_tokens.bin"]["tokens"],
        "holdout_tokens": doc["files"]["holdout_tokens.bin"]["tokens"],
        "tokenizer": doc["tokenizer"]["type"],
        "vocab_size": doc["tokenizer"]["vocab_size"],
        "files": sorted(doc["files"]),
    }, ensure_ascii=False, indent=2))
    print("\nverify: python -m autotrain.corpus verify --corpus " + args.out)


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify(args.corpus)
    print(json.dumps({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "train_tokens": doc["files"]["train_tokens.bin"]["tokens"],
        "holdout_tokens": doc["files"]["holdout_tokens.bin"]["tokens"],
        "files": sorted(doc["files"]),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="General-purpose LM corpus pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("fetch", help="download a pinned public text corpus")
    sp.add_argument("--source", default="wiki_zh", choices=sorted(SOURCES))
    sp.add_argument("--sources-dir", default=DEFAULT_SOURCES_DIR)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build", help="build tokenizer + train/holdout token streams")
    sp.add_argument("--source", default="wiki_zh", choices=sorted(SOURCES))
    sp.add_argument("--sources-dir", default=DEFAULT_SOURCES_DIR)
    sp.add_argument("--out", default=DEFAULT_CORPUS_DIR)
    sp.add_argument("--vocab-size", type=int, default=8192)
    sp.add_argument("--tokenizer-kind", default="char", choices=("char", "bpe"))
    sp.add_argument("--holdout-fraction", type=float, default=0.05)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-chars", type=int, default=None,
                    help="cap training chars (tokenizer still uses full text)")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("verify", help="verify checksums and the corpus seal")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS_DIR)
    sp.set_defaults(func=cmd_verify)
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
