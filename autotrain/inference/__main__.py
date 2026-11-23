"""CLI: python -m autotrain.inference {serve,generate,info}

Examples (this Android phone)::

    python -m autotrain.inference info --bundle runs/android-verify/bundle
    python -m autotrain.inference generate --bundle runs/android-verify/bundle \
        --prompt-ids 1,2,3,4 --max-tokens 16
    python -m autotrain.inference serve --bundle runs/android-verify/bundle \
        --host 127.0.0.1 --port 8080 --device cpu --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import GenerationConfig, InferenceEngine, default_threads
from .chat import ChatTemplate
from .tokenizer import IdentityTokenizer, load_tokenizer
from .server import serve


def _bundleEngine(args: argparse.Namespace) -> InferenceEngine:
    return InferenceEngine.from_bundle(
        args.bundle,
        device=args.device,
        threads=args.threads,
        verify=not args.no_verify,
    )


def _prompt_ids(raw: str) -> list[int]:
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        raise SystemExit(f"--prompt-ids must be comma-separated integers: {e}") from e


def cmd_info(args: argparse.Namespace) -> None:
    engine = _bundleEngine(args)
    print(json.dumps(engine.health(), indent=2, ensure_ascii=False, sort_keys=True))


def cmd_generate(args: argparse.Namespace) -> None:
    engine = _bundleEngine(args)
    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    if tok is not None and tok.vocab_size != engine.vocab_size:
        raise SystemExit("tokenizer vocab_size does not match the model")

    if args.prompt is not None:
        if tok is None:
            raise SystemExit("--prompt requires --tokenizer")
        prompt_ids = tok.encode(args.prompt)
    elif args.prompt_ids:
        prompt_ids = _prompt_ids(args.prompt_ids)
    else:
        raise SystemExit("provide --prompt-ids, or --prompt with --tokenizer")

    cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        stop_token_ids=tuple(args.stop_token_ids or ()),
    )
    result = engine.generate(prompt_ids, cfg)
    payload = {"model": engine.model_id(), **result.as_dict()}
    if tok is not None:
        payload["text"] = tok.decode(result.token_ids)
        payload["prompt"] = args.prompt
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"model={engine.model_id()}")
        print(f"prompt={args.prompt if args.prompt is not None else prompt_ids}")
        if tok is not None:
            print(f"text={payload['text']!r}")
        else:
            print(f"generated={result.token_ids}")
        print(f"{result.finish_reason} | {result.completion_tokens} tokens | "
              f"{result.elapsed_sec:.3f}s | cache={result.cache_bytes}B")


def cmd_serve(args: argparse.Namespace) -> None:
    engine = _bundleEngine(args)
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        if tokenizer.vocab_size != engine.vocab_size:
            raise SystemExit(
                f"tokenizer vocab_size {tokenizer.vocab_size} != model vocab_size "
                f"{engine.vocab_size}; rebuild the corpus with the same budget"
            )
    else:
        tokenizer = IdentityTokenizer(engine.vocab_size, eos_token_id=args.eos_token_id)

    chat_template = None
    if args.chat_template:
        chat_template = ChatTemplate.from_file(args.chat_template)
        chat_template.validate_chars(tokenizer)

    retriever = None
    if args.retrieval_index:
        from ..retrieval import Retriever
        retriever = Retriever.from_dir(args.retrieval_index)
        print(f"retrieval: index={args.retrieval_index} entries={len(retriever._entries)}",
              flush=True)

    api_key = args.api_key or os.environ.get("WANXIANG_API_KEY")
    print(json.dumps({
        "health": f"http://{args.host}:{args.port}/health",
        "endpoints": [
            "/v1/token-completions",
            "/v1/completions" + (" (text)" if tokenizer.supports_text else " (400: no tokenizer)"),
            "/v1/chat/completions" + (" (SFT)" if chat_template else " (501: no chat template)"),
        ],
        "tokenizer": tokenizer.type,
        "chat_template": chat_template.template_hash[:16] if chat_template else None,
        "retrieval": bool(retriever),
        "api_key_required": bool(api_key),
        "max_concurrency": args.max_concurrency,
    }, ensure_ascii=False, indent=2), flush=True)
    serve(
        engine, tokenizer, chat_template=chat_template, retriever=retriever,
        host=args.host, port=args.port, api_key=api_key,
        max_concurrency=args.max_concurrency, verbose=args.verbose,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wanxiang inference service (M1 token-id level)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--bundle", required=True, help="M1.5 bundle directory")
        sp.add_argument("--device", default=None,
                        help="cpu / cuda / mps; default = loader's cuda->mps->cpu")
        sp.add_argument("--threads", type=int, default=None,
                        help=f"torch intra-op threads (default {default_threads()})")
        sp.add_argument("--no-verify", action="store_true",
                        help="skip per-tensor checksum verification (faster, unsafe for prod)")

    sp_info = sub.add_parser("info", help="print model / cache budget")
    common(sp_info)
    sp_info.set_defaults(func=cmd_info)

    sp_gen = sub.add_parser("generate", help="one-shot generation from the CLI")
    common(sp_gen)
    sp_gen.add_argument("--prompt-ids", default=None, help="comma-separated token ids")
    sp_gen.add_argument("--prompt", default=None, help="text prompt (needs --tokenizer)")
    sp_gen.add_argument("--tokenizer", default=None, help="M2/M3 tokenizer json (char or bpe)")
    sp_gen.add_argument("--max-tokens", type=int, default=16)
    sp_gen.add_argument("--do-sample", action="store_true")
    sp_gen.add_argument("--temperature", type=float, default=1.0)
    sp_gen.add_argument("--top-k", type=int, default=0)
    sp_gen.add_argument("--top-p", type=float, default=1.0)
    sp_gen.add_argument("--seed", type=int, default=0)
    sp_gen.add_argument("--stop-token-ids", type=int, nargs="*", default=[])
    sp_gen.add_argument("--json", action="store_true")
    sp_gen.set_defaults(func=cmd_generate)

    sp_serve = sub.add_parser("serve", help="start the HTTP server")
    common(sp_serve)
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8080)
    sp_serve.add_argument("--api-key", default=None,
                          help="or set WANXIANG_API_KEY; required for non-loopback hosts")
    sp_serve.add_argument("--tokenizer", default=None,
                          help="M2/M3 tokenizer json (char or bpe); enables /v1/completions")
    sp_serve.add_argument("--chat-template", default=None,
                          help="SFT versioned chat template json; enables /v1/chat/completions")
    sp_serve.add_argument("--retrieval-index", default=None,
                          help="sealed retrieval index dir; exact 成语 hits answered by lookup")
    sp_serve.add_argument("--eos-token-id", type=int, default=None)
    sp_serve.add_argument("--max-concurrency", type=int, default=1)
    sp_serve.add_argument("--verbose", action="store_true", help="log HTTP requests")
    sp_serve.set_defaults(func=cmd_serve)
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
