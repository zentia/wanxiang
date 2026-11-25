"""Anonymised collector: local source tree -> sealed, path-free source jsonl.

Determinism: candidate paths are sorted by sha256(relative_path + seed), so a
re-run with the same arguments yields byte-identical output regardless of
filesystem enumeration order. Duplicates are removed by content hash. Sampling
is uniform over the whole tree (not alphabetical), so language mix is stable.

The emitted records contain ONLY: cleaned text, language tag, a content hash,
line count and char count. No path, no filename, no provenance string.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# Code-like sources. Config/project files (.xml/.json/.sln/.csproj/.meta) are
# deliberately excluded: they are high-volume, low-signal boilerplate.
CODE_EXTENSIONS = frozenset({
    ".cs", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".m", ".mm",
    ".py", ".lua", ".ts", ".tsx", ".js", ".jsx", ".java", ".kt",
    ".rs", ".go", ".proto", ".shader", ".hlsl", ".glsl", ".cginc",
    ".compute", ".cmake", ".sh", ".bat", ".ps1",
})

# Directory names (lowercased) pruned during the walk. Third-party / generated
# trees would otherwise dominate the sample and leak vendor code.
# Generic build/output/vendor directories. Dot-prefixed directories are
# pruned separately; no environment-specific or project-specific names here.
EXCLUDE_DIRS = frozenset({
    ".git", ".svn", ".hg", ".vs", ".idea", ".vscode", ".gradle",
    ".pytest_cache", "__pycache__", ".cache", ".local",
    "builds", "build", "library", "temp", "obj", "bin", "node_modules",
    "thirdparty", "3rdparty", "vendor", "extern", "external", "packages",
    "deriveddatacache", "intermediate", "backup", "backups", "logs",
    "usersettings", "dist", "out", "target",
})

EXCLUDE_FILE_NAMES = frozenset({
    "assemblyinfo.cs", "package-lock.json", "packages.lock.json",
})
EXCLUDE_SUFFIXES = (
    ".meta", ".min.js", ".map", ".generated.cs", ".g.cs", ".designer.cs",
    ".asset", ".prefab", ".unity", ".dll", ".pdb", ".exe", ".so", ".a",
    ".lib", ".pyc",
)

MIN_BYTES = 200
MAX_FILE_BYTES = 512 * 1024
MIN_LINES = 3
MIN_CHARS = 200
SAMPLING_VERSION = "wanxiang-code-sampling-1"

# ---------------------------------------------------------------------------
# Cleaning: preserve code structure.
# ---------------------------------------------------------------------------


def clean_code(text: str) -> str:
    """Normalise line endings, keep newlines/tabs/indentation, drop junk.

    Unlike ``autotrain.corpus.pipeline.clean_text`` this does NOT flatten
    whitespace: indentation and line breaks ARE the syntax of source code.
    """
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept = [ch for ch in text if ch in ("\n", "\t") or ch.isprintable()]
    text = "".join(kept)

    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse 3+ consecutive blank lines into one; keep single blank lines.
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run <= 1:
                out.append(line)

    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def _decode(raw: bytes) -> str | None:
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _looks_like_text(text: str) -> bool:
    if "\x00" in text:
        return False
    sample = text[:4096]
    if not sample:
        return False
    printable = sum(1 for ch in sample if ch.isprintable() or ch in ("\n", "\t"))
    return printable / len(sample) >= 0.90


# ---------------------------------------------------------------------------
# Candidate walk.
# ---------------------------------------------------------------------------


def _candidate_files(root: Path) -> Iterator[tuple[Path, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in EXCLUDE_DIRS
            and not d.startswith(".")
            and "backup" not in d.lower()
        )
        for name in sorted(filenames):
            lower = name.lower()
            if lower in EXCLUDE_FILE_NAMES or lower.endswith(EXCLUDE_SUFFIXES):
                continue
            p = Path(dirpath) / name
            if p.suffix.lower() not in CODE_EXTENSIONS:
                continue
            yield p, str(p.relative_to(root)).replace("\\", "/")


def _sample_key(rel: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{rel}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Collect.
# ---------------------------------------------------------------------------


def collect(
    root: str | Path,
    out_jsonl: str | Path,
    *,
    max_bytes: int = 32 * 1024 * 1024,
    sample_files: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Write an anonymised source jsonl; return an anonymous summary dict."""
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"source root not found: {root_p}")

    candidates = sorted(
        _candidate_files(root_p), key=lambda item: _sample_key(item[1], seed)
    )
    print(f"[collect] candidate files: {len(candidates)}")

    seen: set[str] = set()
    by_lang: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    docs: list[dict[str, Any]] = []
    total_bytes = 0

    for p, rel in candidates:
        if len(docs) >= sample_files:
            excluded["sample_cap"] += 1
            continue
        if total_bytes >= max_bytes:
            break
        try:
            raw = p.read_bytes()
        except OSError:
            excluded["unreadable"] += 1
            continue
        if len(raw) < MIN_BYTES:
            excluded["too_small"] += 1
            continue
        if len(raw) > MAX_FILE_BYTES:
            excluded["too_large"] += 1
            continue

        text = _decode(raw)
        if text is None:
            excluded["decode"] += 1
            continue
        if not _looks_like_text(text):
            excluded["binary"] += 1
            continue

        cleaned = clean_code(text)
        lines = cleaned.count("\n")
        if lines < MIN_LINES or len(cleaned) < MIN_CHARS:
            excluded["too_short_after_clean"] += 1
            continue

        doc_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if doc_hash in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(doc_hash)

        lang = p.suffix.lower().lstrip(".")
        rec = {
            "text": cleaned,
            "lang": lang,
            "doc_sha256": doc_hash[:16],
            "lines": lines,
            "chars": len(cleaned),
        }
        docs.append(rec)
        by_lang[lang] += 1
        total_bytes += len(cleaned.encode("utf-8"))

    if len(docs) < 10:
        raise RuntimeError(f"only {len(docs)} usable code documents; corpus too small")

    out_p = Path(out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8", newline="\n") as f:
        for rec in docs:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary: dict[str, Any] = {
        "sampling_version": SAMPLING_VERSION,
        "seed": seed,
        "max_bytes": max_bytes,
        "sample_files": sample_files,
        "candidates": len(candidates),
        "documents": len(docs),
        "bytes": total_bytes,
        "by_lang": dict(sorted(by_lang.items())),
        "excluded": dict(sorted(excluded.items())),
        "source_sha256": hashlib.sha256(out_p.read_bytes()).hexdigest(),
        "note": "Anonymised local source snapshot. Paths and filenames are "
                "deliberately omitted; this file is a local artifact under runs/.",
    }
    (out_p.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
