"""CLI: python -m autotrain.code_corpus {collect,build,verify,stats}

    python -m autotrain.code_corpus collect --root <SOURCE_ROOT> \
        --out runs/code-corpus/sources/code.jsonl
    python -m autotrain.code_corpus build \
        --source runs/code-corpus/sources/code.jsonl --out runs/code-corpus
    python -m autotrain.code_corpus verify --corpus runs/code-corpus
    python -m autotrain.corpus verify --corpus runs/code-corpus   # same seal
    python -m autotrain.eval_lm score --bundle <B> \
        --corpus runs/code-corpus --out report.json --seq-len 512 --batch-size 8

    # Dialogue layer (answer-side completion SFT)
    python -m autotrain.code_corpus sft-build \
        --source runs/code-corpus/sources/code.jsonl \
        --tokenizer runs/code-corpus/tokenizer.json --out runs/code-sft
    python -m autotrain.code_corpus sft-verify --corpus runs/code-sft
    python -m autotrain.code_corpus sft-score --bundle <B> \
        --corpus runs/code-sft --seq-len 512 --batch-size 8

All outputs are local derived artifacts under ``runs/`` (gitignored). Never
add source text, the jsonl, token streams or manifests to git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import build, verify
from .collect import collect
from .sft import build_sft, score_sft, verify_sft

DEFAULT_SOURCE = "runs/code-corpus/sources/code.jsonl"
DEFAULT_CORPUS = "runs/code-corpus"


def cmd_collect(args: argparse.Namespace) -> None:
    summary = collect(
        args.root, args.out,
        max_bytes=args.max_bytes, sample_files=args.sample_files, seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    doc = build(
        args.source, args.out,
        vocab_size=args.vocab_size, tokenizer_kind=args.tokenizer_kind,
        holdout_fraction=args.holdout_fraction, seed=args.seed,
        tokenizer_sample_chars=args.tokenizer_sample_chars,
    )
    print(json.dumps({
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "documents": doc["extraction"]["documents"],
        "by_lang": doc["extraction"]["by_lang"],
        "train_tokens": doc["files"]["train_tokens.bin"]["tokens"],
        "holdout_tokens": doc["files"]["holdout_tokens.bin"]["tokens"],
        "tokenizer": doc["tokenizer"]["type"],
        "vocab_size": doc["tokenizer"]["vocab_size"],
        "n_merges": doc["tokenizer"]["n_merges"],
        "note": "verify: python -m autotrain.code_corpus verify --corpus " + args.out,
    }, ensure_ascii=False, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify(args.corpus)
    print(json.dumps({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "train_tokens": doc["files"]["train_tokens.bin"]["tokens"],
        "holdout_tokens": doc["files"]["holdout_tokens.bin"]["tokens"],
        "tokenizer": doc["tokenizer"]["type"],
    }, ensure_ascii=False, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    p = Path(args.corpus) / "sources" / "summary.json"
    if not p.is_file():
        p = Path(args.source).parent / "summary.json"
    print(p.read_text(encoding="utf-8"))


def cmd_sft_build(args: argparse.Namespace) -> None:
    doc = build_sft(
        args.source, args.tokenizer, args.out,
        holdout_fraction=args.holdout_fraction, seed=args.seed,
        prefix_lines=args.prefix_lines, answer_lines=args.answer_lines,
        stride=args.stride, max_seq_len=args.max_seq_len,
        max_samples=args.max_samples, mode=args.mode,
    )
    print(json.dumps({
        "corpus_id": doc["corpus_id"],
        "version": doc["version"],
        "samples": doc["extraction"]["samples"],
        "by_lang": doc["extraction"]["by_lang"],
        "template_hash": doc["template"]["hash"],
        "train_tokens": doc["files"]["sft_train_tokens.bin"]["tokens"],
        "holdout_tokens": doc["files"]["sft_holdout_tokens.bin"]["tokens"],
        "note": "verify: python -m autotrain.code_corpus sft-verify --corpus " + args.out,
    }, ensure_ascii=False, indent=2))


def cmd_sft_verify(args: argparse.Namespace) -> None:
    doc = verify_sft(args.corpus)
    print(json.dumps({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "samples": doc["extraction"]["samples"],
        "template_hash": doc["template"]["hash"],
    }, ensure_ascii=False, indent=2))


def cmd_sft_score(args: argparse.Namespace) -> None:
    result = score_sft(
        args.bundle, args.corpus,
        seq_len=args.seq_len, batch_size=args.batch_size,
        device=args.device, max_windows=args.max_windows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Anonymised local code-LM corpus pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("collect", help="scan a local source tree -> anonymised jsonl")
    sp.add_argument("--root", required=True, help="local source root (never recorded)")
    sp.add_argument("--out", default=DEFAULT_SOURCE)
    sp.add_argument("--max-bytes", type=int, default=32 * 1024 * 1024,
                    help="cap on cleaned corpus bytes (default 32 MiB)")
    sp.add_argument("--sample-files", type=int, default=20000)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("build", help="jsonl -> frozen train/holdout streams")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--out", default=DEFAULT_CORPUS)
    sp.add_argument("--vocab-size", type=int, default=8192)
    sp.add_argument("--tokenizer-kind", default="bpe", choices=("bpe", "char"))
    sp.add_argument("--holdout-fraction", type=float, default=0.05)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--tokenizer-sample-chars", type=int,
                    default=4 * 1024 * 1024)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("verify", help="re-check the corpus seal")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("stats", help="print the anonymous collection summary")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS)
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("sft-build", help="code completion instruction pairs")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--tokenizer", default="runs/code-corpus/tokenizer.json")
    sp.add_argument("--out", default="runs/code-sft")
    sp.add_argument("--holdout-fraction", type=float, default=0.05)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--prefix-lines", type=int, default=8)
    sp.add_argument("--answer-lines", type=int, default=12)
    sp.add_argument("--stride", type=int, default=32)
    sp.add_argument("--max-seq-len", type=int, default=512)
    sp.add_argument("--max-samples", type=int, default=200000)
    sp.add_argument("--mode", default="window", choices=("window", "block"),
                    help="window = legacy fixed lines; block = brace-aligned, "
                         "template-filtered, answer-deduplicated")
    sp.set_defaults(func=cmd_sft_build)

    sp = sub.add_parser("sft-verify", help="re-check the SFT corpus seal")
    sp.add_argument("--corpus", default="runs/code-sft")
    sp.set_defaults(func=cmd_sft_verify)

    sp = sub.add_parser("sft-score", help="answer-side CE / top-1 on SFT holdout")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--corpus", default="runs/code-sft")
    sp.add_argument("--seq-len", type=int, default=512)
    sp.add_argument("--batch-size", type=int, default=8)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--max-windows", type=int, default=400)
    sp.set_defaults(func=cmd_sft_score)
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
