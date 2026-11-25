"""CLI: python -m autotrain.eval_gen {score,compare,verify}

    # frozen ruler, decode controls OFF (historical greedy behaviour)
    python -m autotrain.eval_gen score --bundle runs/code-sft-train-v2/bundle \
        --corpus runs/code-corpus --out runs/eval_gen/off.json

    # same ruler, decode controls ON
    python -m autotrain.eval_gen score --bundle runs/code-sft-train-v2/bundle \
        --corpus runs/code-corpus --out runs/eval_gen/on.json \
        --repetition-penalty 1.2 --no-repeat-ngram 3 --frequency-penalty 0.1

    python -m autotrain.eval_gen compare --baseline off.json --candidate on.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import DEFAULT_N_PROMPTS, GEN_LEN, freeze_prompts, compare, score


def cmd_score(args: argparse.Namespace) -> None:
    decode = {
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
        "frequency_penalty": args.frequency_penalty,
    }
    report = score(
        args.bundle, args.corpus, decode=decode,
        device=args.device, threads=args.threads,
        n_prompts=args.n_prompts, gen_len=args.gen_len,
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
        "bundle_step": report["bundle_step"],
        "decode_config": report["decode_config"],
        "metrics": {k: round(v, 4) for k, v in report["metrics"].items()},
    }, ensure_ascii=False, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(compare(before, after), ensure_ascii=False, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    p = freeze_prompts(args.corpus, n=args.n_prompts)
    print(json.dumps({
        "status": "OK",
        "corpus_id": p["corpus_id"],
        "prompt_sha256": p["sha256"],
        "n_prompts": len(p["prompts"]),
        "starts": p["starts"][:5],
        "note": "deterministic prompt set re-derived from the frozen holdout",
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frozen objective generation-quality (repetition) ruler"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("score", help="greedy-generate on frozen prompts")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--corpus", default="runs/code-corpus")
    sp.add_argument("--out", required=True)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    sp.add_argument("--gen-len", type=int, default=GEN_LEN)
    sp.add_argument("--repetition-penalty", type=float, default=1.0)
    sp.add_argument("--no-repeat-ngram", type=int, default=0)
    sp.add_argument("--frequency-penalty", type=float, default=0.0)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("compare", help="strict same-ruler A/B")
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("verify", help="re-derive the frozen prompt set")
    sp.add_argument("--corpus", default="runs/code-corpus")
    sp.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
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
