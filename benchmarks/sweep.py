"""Automated serving-parameter sweep for llama.cpp (Windows / CUDA).

Restarts the server once per (parallel slots, context) configuration,
prewarms (including a batch at the target concurrency), then runs the same
load level and records throughput, latency distribution and VRAM.

Why a sweep instead of a single run: slot count and context length are the
two knobs that set the KV-cache budget and the scheduling capacity. Sweeping
them produces the capacity curve an inference platform team actually needs.

    python benchmarks/sweep.py --out benchmarks/results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from .serving_bench import one_request, run_level, vram_mib

SERVER = "runs/llamacpp-cuda/llama-server.exe"
MODEL = "runs/qwen-gguf/qwen-7b-Q4_K_M.gguf"
HOST, PORT = "127.0.0.1", 8083
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
HEALTH = f"http://{HOST}:{PORT}/health"

# (parallel slots, total context) -- deliberately spans both dimensions.
CONFIGS: list[tuple[int, int]] = [
    (4, 2048),
    (8, 4096),
    (16, 8192),
    (8, 16384),
    (16, 16384),
    (32, 8192),
]


def wait_health(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(slots: int, ctx: int) -> subprocess.Popen:
    cmd = [
        SERVER, "-m", MODEL,
        "--host", HOST, "--port", str(PORT),
        "-c", str(ctx), "-np", str(slots),
        "--n-gpu-layers", "99",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()


def run_config(slots: int, ctx: int, concurrency: int,
               max_tokens: int, timeout: int) -> dict[str, Any]:
    proc = start_server(slots, ctx)
    try:
        if not wait_health():
            return {"slots": slots, "ctx": ctx, "error": "health timeout"}
        # Two-layer prewarm: absorb PTX JIT and the batch-variant cold path.
        one_request(URL, 8, timeout)
        run_level(URL, concurrency, 8, timeout)
        before = vram_mib()
        result = run_level(URL, concurrency, max_tokens, timeout)
        after = vram_mib()
        result.update({
            "slots": slots,
            "ctx": ctx,
            "ctx_per_slot": ctx // slots,
            "vram_before_mib": before,
            "vram_after_mib": after,
        })
        return result
    finally:
        stop_server(proc)
        time.sleep(3)  # let the CUDA context tear down / VRAM release


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="benchmarks/results/sweep.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for slots, ctx in CONFIGS:
        print(f"--- slots={slots} ctx={ctx} (per slot {ctx // slots}) ---",
              flush=True)
        r = run_config(slots, ctx, args.concurrency, args.max_tokens,
                       args.timeout)
        rows.append(r)
        if "error" in r:
            print(f"  FAILED: {r['error']}", flush=True)
        else:
            print(
                f"  TTFT={r['ttft_mean_ms']}ms TPOT={r['tpot_mean_ms']}ms "
                f"p99={r['e2e_p99_ms']}ms agg={r['aggregate_tok_per_sec']} tok/s "
                f"vram={r['vram_after_mib']['used']} MiB "
                f"ok={r['ok']}/{r['requests']}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "configs": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
