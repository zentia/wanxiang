"""CLI: python -m autotrain.qwen_sft {build,verify,train}

    python -m autotrain.qwen_sft build \
        --base runs/qwen-base/1.5b \
        --source runs/code-corpus-v2/sources/code.jsonl \
        --out runs/qwen-sft --max-samples 20000
    python -m autotrain.qwen_sft verify --data runs/qwen-sft
    python -m autotrain.qwen_sft train --base runs/qwen-base/1.5b \
        --data runs/qwen-sft --out runs/qwen-sft/run --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapt import compare as adapt_compare, score as adapt_score
from .data import build, verify
from .quality import compare as quality_compare, score as quality_score
from .ruler import compare as gen_compare, score as gen_score
from .serve import run as serve_run
from .syntax import compare as syntax_compare, score as syntax_score
from .train import run


def _print(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_build(args: argparse.Namespace) -> None:
    doc = build(
        args.base, args.source, args.out,
        max_samples=args.max_samples, max_seq_len=args.max_seq_len,
        holdout_fraction=args.holdout_fraction, seed=args.seed,
    )
    _print({
        "corpus_id": doc["corpus_id"],
        "base": doc["base"],
        "tokenizer_vocab_size": doc["tokenizer_vocab_size"],
        "samples": doc["extraction"]["samples"],
        "by_lang": doc["extraction"]["by_lang"],
    })


def cmd_verify(args: argparse.Namespace) -> None:
    doc = verify(args.data)
    _print({
        "status": "OK",
        "corpus_id": doc["corpus_id"],
        "samples": doc["extraction"]["samples"],
        "files": {k: v["rows"] for k, v in doc["files"].items()},
    })


def cmd_train(args: argparse.Namespace) -> None:
    _print(run(
        args.base, args.data, args.out,
        max_steps=args.max_steps, max_samples=args.max_samples,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ))


def cmd_score(args: argparse.Namespace) -> None:
    decode = {
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    report = gen_score(
        args.base, args.source,
        adapter_dir=args.adapter,
        decode=decode,
        n_prompts=args.n_prompts,
        gen_len=args.gen_len,
        load_in_4bit=not args.full_precision,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out.resolve()),
        "gen_hash": report["gen_hash"],
        "base": report["base"],
        "adapter": report["adapter"],
        "metrics": {k: round(v, 4) for k, v in report["metrics"].items()},
    }, ensure_ascii=False, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(gen_compare(before, after), ensure_ascii=False, indent=2))


def cmd_adapt_score(args: argparse.Namespace) -> None:
    report = adapt_score(
        args.base, args.data,
        adapter_dir=args.adapter,
        n=args.n,
        load_in_4bit=not args.full_precision,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out.resolve()),
        "adapt_hash": report["adapt_hash"],
        "adapter": report["adapter"],
        "answer_cross_entropy": round(report["answer_cross_entropy"], 4),
        "answer_top1_accuracy": round(report["answer_top1_accuracy"], 4),
        "answer_tokens": report["answer_tokens"],
    }, ensure_ascii=False, indent=2))


def cmd_adapt_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(adapt_compare(before, after), ensure_ascii=False, indent=2))


def cmd_quality_score(args: argparse.Namespace) -> None:
    decode = {
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    report = quality_score(
        args.base, args.data,
        adapter_dir=args.adapter,
        decode=decode,
        n=args.n,
        gen_len=args.gen_len,
        load_in_4bit=not args.full_precision,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out.resolve()),
        "quality_hash": report["quality_hash"],
        "base": report["base"],
        "adapter": report["adapter"],
        "metrics": {k: round(v, 4) for k, v in report["metrics"].items()},
    }, ensure_ascii=False, indent=2))


def cmd_quality_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(quality_compare(before, after), ensure_ascii=False, indent=2))


def cmd_syntax_score(args: argparse.Namespace) -> None:
    decode = {
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    report = syntax_score(
        args.base, args.data,
        adapter_dir=args.adapter,
        decode=decode,
        n=args.n,
        gen_len=args.gen_len,
        load_in_4bit=not args.full_precision,
        checker=args.checker,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out.resolve()),
        "syntax_hash": report["syntax_hash"],
        "base": report["base"],
        "adapter": report["adapter"],
        "metrics": {k: round(v, 4) for k, v in report["metrics"].items()},
    }, ensure_ascii=False, indent=2))


def cmd_syntax_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(syntax_compare(before, after), ensure_ascii=False, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    serve_run(
        args.base,
        adapter_dir=args.adapter,
        host=args.host,
        port=args.port,
        max_new_tokens=args.max_tokens,
        api_key=args.api_key,
        load_in_4bit=not args.full_precision,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA adaptation of a Qwen code base")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build", help="re-encode local SFT data with base tokenizer")
    sp.add_argument("--base", required=True)
    sp.add_argument("--source", default="runs/code-corpus-v2/sources/code.jsonl")
    sp.add_argument("--out", default="runs/qwen-sft")
    sp.add_argument("--max-samples", type=int, default=20000)
    sp.add_argument("--max-seq-len", type=int, default=1024)
    sp.add_argument("--holdout-fraction", type=float, default=0.05)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("verify", help="re-check the sealed SFT data")
    sp.add_argument("--data", default="runs/qwen-sft")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("train", help="4-bit QLoRA training")
    sp.add_argument("--base", required=True)
    sp.add_argument("--data", default="runs/qwen-sft")
    sp.add_argument("--out", default="runs/qwen-sft/run")
    sp.add_argument("--max-steps", type=int, default=20)
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--lora-r", type=int, default=16)
    sp.add_argument("--lora-alpha", type=int, default=32)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("score", help="objective generation ruler on frozen prompts")
    sp.add_argument("--base", required=True)
    sp.add_argument("--source", default="runs/code-corpus-v2/sources/code.jsonl")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n-prompts", type=int, default=16)
    sp.add_argument("--gen-len", type=int, default=64)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--no-repeat-ngram", type=int, default=0)
    sp.add_argument("--full-precision", action="store_true",
                    help="load base in bf16 instead of 4-bit")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("compare", help="strict same-ruler A/B")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("adapt-score", help="answer-side CE on frozen holdout")
    sp.add_argument("--base", required=True)
    sp.add_argument("--data", default="runs/qwen-sft")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--full-precision", action="store_true")
    sp.set_defaults(func=cmd_adapt_score)

    sp = sub.add_parser("adapt-compare", help="same-ruler adaptation A/B")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.set_defaults(func=cmd_adapt_compare)

    sp = sub.add_parser("quality-score", help="format / correctness-proxy ruler")
    sp.add_argument("--base", required=True)
    sp.add_argument("--data", default="runs/qwen-sft")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=32)
    sp.add_argument("--gen-len", type=int, default=64)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--no-repeat-ngram", type=int, default=0)
    sp.add_argument("--full-precision", action="store_true")
    sp.set_defaults(func=cmd_quality_score)

    sp = sub.add_parser("quality-compare", help="same-ruler format/quality A/B")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.set_defaults(func=cmd_quality_compare)

    sp = sub.add_parser("syntax-score", help="Roslyn syntax ruler")
    sp.add_argument("--base", required=True)
    sp.add_argument("--data", default="runs/qwen-sft")
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=32)
    sp.add_argument("--gen-len", type=int, default=64)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--no-repeat-ngram", type=int, default=0)
    sp.add_argument("--full-precision", action="store_true")
    sp.add_argument("--checker", default=None,
                    help="path to SyntaxCheck.exe; auto-built if missing")
    sp.set_defaults(func=cmd_syntax_score)

    sp = sub.add_parser("syntax-compare", help="same-ruler syntax A/B")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.set_defaults(func=cmd_syntax_compare)

    sp = sub.add_parser("serve", help="local OpenAI-shaped HTTP server")
    sp.add_argument("--base", required=True)
    sp.add_argument("--adapter", default=None)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--max-tokens", type=int, default=256)
    sp.add_argument("--api-key", default=None)
    sp.add_argument("--full-precision", action="store_true")
    sp.set_defaults(func=cmd_serve)
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
