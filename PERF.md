# Inference Serving Performance Report

> SUT: `Qwen2.5-Coder-7B-Instruct` + 400-step QLoRA adapter, exported as
> `Q4_K_M` GGUF (4.36 GB) and served by llama.cpp build `b12359` (CUDA 12.4).
> Framed the way a real-time / engine performance report is framed: **budgets,
> jitter, cold paths and capacity limits**, not model quality.

## 1. System under test

| | |
|---|---|
| GPU | RTX 5080 16 GB, driver 617.42, compute capability 12.0 |
| Runtime | CUDA 12.4 build + matching `cudart64_12.dll` / `cublas64_12.dll` |
| Model | 7.62 B params, Q4_K_M (4.91 BPW), 4.36 GiB |
| Server | `llama-server`, `--n-gpu-layers 99`, `-c 8192`, `-np 16` |
| API | OpenAI-compatible `/v1/chat/completions`, SSE streaming |
| Workload | 49-token prompt, 32 generated tokens, greedy (`temperature=0`) |

Fixed context 8192 split across 16 slots = 512 ctx/slot, enough for the
workload. KV cache is **preallocated at startup**, not grown on demand.

## 2. Method

[benchmarks/serving_bench.py](<benchmarks/serving_bench.py>) fires a
concurrency ladder `1 / 4 / 8 / 16 / 32` with all requests in flight at once,
reads SSE chunks, and records:

- **TTFT** — time to first content token (true streaming measurement)
- **TPOT** — mean inter-token interval
- **e2e p50 / p95 / p99 / max** — end-to-end latency distribution
- **aggregate tok/s** — total generated tokens / wall time
- **VRAM** — `nvidia-smi` before and after each level
- **error rate**

Two rounds were run back-to-back on the same warm server: round 1 with two
warm-up requests, round 2 with none. Raw data:
[round 1](<benchmarks/results/serving_bench.json>),
[round 2](<benchmarks/results/serving_bench_run2.json>).

## 3. Steady-state results (round 2)

| c | ok | TTFT ms | TPOT ms | e2e p50 | p95 | p99 | max | agg tok/s | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/1 | 30.6 | 6.5 | 89.7 | 89.7 | 89.7 | 89.7 | 99.7 | 8188 |
| 4 | 4/4 | 32.9 | 8.3 | 116.0 | 116.1 | 116.1 | 116.1 | 308.5 | 8188 |
| 8 | 8/8 | 22.1 | 11.1 | 115.5 | 136.1 | 136.1 | 136.1 | 525.6 | 8188 |
| 16 | 16/16 | 20.3 | 15.0 | 155.3 | 157.5 | 157.5 | 157.5 | **889.9** | 8187 |
| 32 | 32/32 | 104.1 | 14.8 | 308.2 | 320.4 | 324.7 | 324.7 | 858.7 | 8187 |

Zero errors at every level. `p99 ≈ max` at all levels — **no long-tail jitter**.

## 4. Three findings

### 4.1 Throughput scales near-linearly until slot saturation

TPOT rises from 6.5 → 15.0 ms as concurrency grows (each individual stream
gets slower), but aggregate throughput rises **99.7 → 889.9 tok/s (8.9×)**.
This is the whole point of parallel decode slots / continuous batching:
individual latency is traded for system throughput.

At `c=32` with only `-np 16` slots, half the requests queue: TTFT jumps
20.3 → **104.1 ms** and throughput falls back to 858.7 tok/s. **The slot
count is the hard capacity limit**, not GPU compute.

### 4.2 Cold paths exist at two levels, not one

| | round 1 (cold) | round 2 (warm) |
|---|---|---|
| first request of the process | 37 s (PTX JIT on sm_120, see CODE_LM 9.12) | — |
| `c=4` TTFT | **2117.7 ms** | **32.9 ms** |
| `c=4` p99 | 2193.9 ms | 116.1 ms |
| `c=4` aggregate | 16.41 tok/s | 308.5 tok/s |

The first time a new batch size is used, the CUDA graph / slot path is
specialized — a **per-variant cold start**, not a steady-state bottleneck.
This is exactly the engine-side *shader variant / PSO compilation stutter*
pattern: the first frame using a new permutation stalls, then it is cached.

**Implication**: a single warm-up request is not enough. Production prewarming
must replay the **expected concurrency ladder** (and the expected batch
shapes), otherwise the first real burst at an unseen concurrency eats a
multi-second stall.

### 4.3 VRAM is a preallocated pool, not a function of concurrency

VRAM stayed **8171 → 8188 MiB (+17 MiB)** across `c=1 → 32`. The KV cache is
reserved for all 16 slots at startup, so serving more requests does not grow
memory — it only changes latency/throughput. Raising `-np` (or `-c`) is what
moves the memory budget, not the traffic.

For an engine background this is the familiar resource-pool model: budget
capacity up front, then fight over latency, not allocation.

## 5. Budget / capacity guidance

| Target | Setting | Expected |
|---|---|---|
| Low latency, single user | `-np 1` | TTFT ~30 ms, ~100 tok/s |
| Balanced (this run) | `-np 16 -c 8192` | ~890 tok/s, TTFT 20–33 ms, 8.2 GB VRAM |
| More concurrency | `-np 32` | KV cache roughly doubles → re-check VRAM budget before promising |
| Any CUDA 12.4 on sm_120 | prewarm ladder | avoid the 37 s JIT + 2.1 s variant stall |

Latency budget reference: at c≤16, p99 stays under **160 ms**; at c=32 the
queue pushes p99 to **325 ms** — a 2× jump for a 2× concurrency increase,
i.e. the knee of the curve.

## 6. Context: CPU vs GPU (single stream, llama-bench)

`llama-bench -p 128 -n 32 -r 2`, same GGUF:

| | CPU (`-ngl 0`) | CUDA (`-ngl 99`) | ratio |
|---|---|---|---|
| prompt eval | 290.19 t/s | 6755.11 t/s | 23× |
| token gen | 9.26 t/s | 169.64 t/s | 18× |

The single-stream `tg` number (169.64) is consistent with the measured
`c=1` TPOT of 6.5 ms (~154 tok/s); the gap to 889.9 tok/s at c=16 is the
parallel-slot gain.

## 7. Reproduce

```powershell
# serve (16 slots, 8192 ctx, full offload)
.\runs\llamacpp-cuda\llama-server.exe -m runs/qwen-gguf/qwen-7b-Q4_K_M.gguf `
    --host 127.0.0.1 --port 8082 -c 8192 -np 16 --n-gpu-layers 99
# prewarm the concurrency ladder, then benchmark
.\.venv\Scripts\python.exe benchmarks/serving_bench.py `
    --url http://127.0.0.1:8082/v1/chat/completions `
    --concurrency 1,4,8,16,32 --max-tokens 32 --warmup 2 `
    --out runs/perf/serving_bench.json
```

## 8. What this report does NOT claim

TTFT/TPOT here are end-to-end over loopback HTTP with SSE parsing and Python
thread scheduling; they are **not** kernel-level timings. No nsys/ncu
profiling, no vLLM continuous-batching comparison, no power/thermal control.
Those are the next measurement layers.
