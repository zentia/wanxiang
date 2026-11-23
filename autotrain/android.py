"""Android / proot runtime adaptation helpers.

This module intentionally does **not** change training or evaluation
semantics: ``config_hash`` and ``evalspec_hash`` stay byte-identical to
desktop runs. It only

* detects the Android (aarch64, proot) environment,
* reports which torch backend is actually usable,
* caps intra-op CPU threads to a number that behaves well on ARM
  big.LITTLE SoCs instead of spawning one thread per core.

Usage::

    python -m autotrain.android                 # print an environment report
    python -m autotrain.android --apply        # also cap torch CPU threads
    python -m autotrain.android --threads 2

In a script::

    from autotrain.android import apply_cpu_defaults, env_summary
    apply_cpu_defaults()
    print(env_summary()["device"])
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

# ARM phones usually have a few big cores + several little cores. Using all
# of them for small GEMMs is slower and hotter than using the big cluster.
DEFAULT_THREADS = 4


def _proc_version() -> str:
    try:
        return Path("/proc/version").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _mem_total_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def is_android() -> bool:
    """True when running under an Android kernel (proot/Termux included)."""
    return "android" in _proc_version().lower() or bool(os.environ.get("ANDROID_ROOT"))


def is_aarch64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def resolve_threads(threads: int | None = None) -> int:
    """Pick a safe intra-op thread count, bounded by available cores."""
    raw = threads if threads is not None else os.environ.get("WANXIANG_THREADS")
    n = int(raw) if raw else DEFAULT_THREADS
    return max(1, min(n, os.cpu_count() or 1))


def apply_cpu_defaults(threads: int | None = None) -> int:
    """Set torch's intra-op parallelism. Returns the applied thread count.

    ``set_num_interop_threads`` is deliberately not touched: it may only be
    called once per process and raises once torch has done parallel work.
    """
    import torch

    n = resolve_threads(threads)
    torch.set_num_threads(n)
    # Environment variables are set too, so child processes and OpenMP-aware
    # libraries (numpy / OpenBLAS) inherit the same limit.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(n))
    return n


def _usable_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def env_summary() -> dict[str, Any]:
    """A small, dependency-light report of this phone's runtime."""
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "kernel": _proc_version(),
        "android": is_android(),
        "aarch64": is_aarch64(),
        "cpu_count": os.cpu_count(),
        "mem_total_mb": _mem_total_mb(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "recommended_threads": resolve_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": _usable_device(),
    }
    notes: list[str] = []
    if is_android():
        notes.append("Android kernel: only the CPU backend is usable; never pip-install CUDA wheels.")
    if not is_aarch64():
        notes.append("Not aarch64: the +cpu aarch64 wheel is unnecessary here.")
    if info["device"] != "cpu":
        notes.append("Non-CPU backend detected: A/B reports must still pin the same --device.")
    if info["torch_threads"] > info["recommended_threads"]:
        notes.append("Run with --apply to cap intra-op threads for ARM big.LITTLE.")
    info["notes"] = notes
    return info


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wanxiang Android runtime report / tuning")
    parser.add_argument("--apply", action="store_true", help="cap torch CPU threads")
    parser.add_argument("--threads", type=int, default=None, help=f"override thread count (default {DEFAULT_THREADS})")
    args = parser.parse_args()

    if args.apply:
        apply_cpu_defaults(args.threads)
    print(json.dumps(env_summary(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
