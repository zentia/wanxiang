"""Concurrent serving benchmark for an OpenAI-compatible llama.cpp server.

Measures what a real-time / frame-budget mindset cares about: TTFT, TPOT,
p50/p95/p99 end-to-end latency, aggregate throughput, VRAM and error rate
across a concurrency ladder. Uses streaming SSE so first-token time is real
rather than inferred from a non-streaming response.

    python benchmarks/serving_bench.py \
        --url http://127.0.0.1:8082/v1/chat/completions \
        --concurrency 1,4,8,16,32 --max-tokens 32 \
        --out runs/perf/serving_bench.json

The server under test must already be running. Warm up once before measuring
(CUDA builds may JIT-compile kernels on the first request).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROMPT = (
    "complete the code below (cs):\n"
    "public void Update() {\n"
    "    if (m_Texture == null)"
)


def vram_mib() -> dict[str, int | None]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        used, free = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
        return {"used": used, "free": free}
    except Exception:
        return {"used": None, "free": None}


def one_request(url: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    token_times: list[float] = []
    err: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    token_times.append(time.perf_counter())
                if choices[0].get("finish_reason"):
                    break
    except Exception as e:  # noqa: BLE001 - benchmark must record failures
        err = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    return {
        "ok": err is None,
        "error": err,
        "e2e": e2e,
        "ttft": token_times[0] - t0 if token_times else None,
        "n_tokens": len(token_times),
        "tpot": ((token_times[-1] - token_times[0]) / (len(token_times) - 1)
                 if len(token_times) > 1 else None),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def run_level(url: str, c: int, max_tokens: int, timeout: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        futures = [ex.submit(one_request, url, max_tokens, timeout)
                   for _ in range(c)]
        rows = [f.result() for f in as_completed(futures)]
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r["ok"]]
    e2es = [r["e2e"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    total_tokens = sum(r["n_tokens"] for r in ok)
    return {
        "concurrency": c,
        "requests": len(rows),
        "ok": len(ok),
        "errors": [r["error"] for r in rows if not r["ok"]][:5],
        "wall_sec": round(wall, 3),
        "total_tokens": total_tokens,
        "aggregate_tok_per_sec": round(total_tokens / wall, 2) if wall else None,
        "ttft_mean_ms": round(1000 * statistics.mean(ttfts), 1) if ttfts else None,
        "tpot_mean_ms": round(1000 * statistics.mean(tpots), 1) if tpots else None,
        "e2e_p50_ms": round(1000 * percentile(e2es, 0.50), 1) if e2es else None,
        "e2e_p95_ms": round(1000 * percentile(e2es, 0.95), 1) if e2es else None,
        "e2e_p99_ms": round(1000 * percentile(e2es, 0.99), 1) if e2es else None,
        "e2e_max_ms": round(1000 * max(e2es), 1) if e2es else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8082/v1/chat/completions")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="runs/perf/serving_bench.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(f"warming up ({args.warmup} requests) ...", flush=True)
    for _ in range(args.warmup):
        r = one_request(args.url, 8, args.timeout)
        print(f"  warmup ok={r['ok']} e2e={r['e2e']:.2f}s", flush=True)

    results = []
    for c in levels:
        before = vram_mib()
        level = run_level(args.url, c, args.max_tokens, args.timeout)
        after = vram_mib()
        level["vram_before_mib"] = before
        level["vram_after_mib"] = after
        results.append(level)
        print(
            f"c={c:2d} ok={level['ok']}/{level['requests']} "
            f"TTFT={level['ttft_mean_ms']}ms TPOT={level['tpot_mean_ms']}ms "
            f"p99={level['e2e_p99_ms']}ms agg={level['aggregate_tok_per_sec']} tok/s "
            f"vram={after['used']} MiB",
            flush=True,
        )

    report = {
        "url": args.url,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "levels": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
