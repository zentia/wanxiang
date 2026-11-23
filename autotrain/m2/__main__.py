"""CLI: python -m autotrain.m2 {fetch,build,verify,vocab}

    python -m autotrain.m2 build                  # 下载(可幂等) + 构建语料
    python -m autotrain.m2 verify                # 逐文件 checksum + 封印校验
    python -m autotrain.m2 vocab --top 20        # 看词表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..inference.tokenizer import CharTokenizer
from .pipeline import build, verify
from .source import SOURCES, fetch

DEFAULT_SOURCES_DIR = "runs/m2/sources"
DEFAULT_CORPUS_DIR = "runs/m2/corpus"


def cmd_fetch(args: argparse.Namespace) -> None:
    meta = fetch(args.source, args.sources_dir, force=args.force)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    meta = fetch(args.source, args.sources_dir)  # cached -> no download
    doc = build(
        meta["path"], args.out,
        source_meta=meta,
        vocab_size=args.vocab_size,
        max_chars=args.max_chars,
    )
    print(json.dumps({
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "out": str(Path(args.out).resolve()),
        "source": doc["source"]["license"],
        "entries": doc["extraction"]["entries"],
        "text_chars": doc["extraction"]["text_chars"],
        "tokens": doc["files"]["corpus_tokens.bin"]["tokens"],
        "used_chars": doc["tokenizer"]["used_chars"],
        "vocab_size": doc["tokenizer"]["vocab_size"],
        "files": sorted(doc["files"]),
    }, ensure_ascii=False, indent=2))
    print("\nverify: python -m autotrain.m2 verify --corpus " + args.out)


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify(args.corpus)
    print(json.dumps({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "tokens": doc["files"]["corpus_tokens.bin"]["tokens"],
        "used_chars": doc["tokenizer"]["used_chars"],
        "files": sorted(doc["files"]),
    }, ensure_ascii=False, indent=2))


def cmd_vocab(args: argparse.Namespace) -> None:
    tok = CharTokenizer.from_file(Path(args.corpus) / "tokenizer.json")
    chars = [t for t in tok.id_to_token[4:] if not t.startswith("<unused_")]
    print(json.dumps({
        "vocab_size": tok.vocab_size,
        "used_chars": tok.used_chars,
        "special_tokens": tok.id_to_token[:4],
        "top_chars": chars[:args.top],
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wanxiang M2 minimal data pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("fetch", help="download a pinned public Chinese corpus")
    sp.add_argument("--source", default="idiom", choices=sorted(SOURCES))
    sp.add_argument("--sources-dir", default=DEFAULT_SOURCES_DIR)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build", help="build tokenizer + uint16 stream + manifest")
    sp.add_argument("--source", default="idiom", choices=sorted(SOURCES))
    sp.add_argument("--sources-dir", default=DEFAULT_SOURCES_DIR)
    sp.add_argument("--out", default=DEFAULT_CORPUS_DIR)
    sp.add_argument("--vocab-size", type=int, default=8192)
    sp.add_argument("--max-chars", type=int, default=None,
                    help="only write the first N characters (vocab still uses full text)")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("verify", help="verify checksums and the corpus seal")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS_DIR)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("vocab", help="inspect the character vocab")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS_DIR)
    sp.add_argument("--top", type=int, default=24)
    sp.set_defaults(func=cmd_vocab)
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
