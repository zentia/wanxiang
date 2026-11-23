"""Deterministic, sealed retrieval index over the pinned idiom dictionary.

The M3 experiments established that character-level generation cannot answer
a *word-level holdout* idiom: the definition never entered training and is not
derivable from the glyphs (``白兔赤乌 -> 多借指时间``).成语释义 is a lookup task,
so the honest fix is retrieval, not a bigger generator.

Index layout (all under one directory, byte-verified by a manifest):
  ``entries.jsonl``   one row per idiom: ``{word, answer, pinyin}``
  ``manifest.json``   version + source sha256 + row count + file checksums

``answer_for_row`` is the SAME function used by SFT labels and eval-3 references,
so a retrieved answer is byte-identical to what training/eval consider correct.
No holdout exclusion is applied on purpose: retrieval is allowed to consult the
whole dictionary -- that is the point of a lookup tool. The eval-4 instrument
states this openly and scores accordingly (see autotrain/eval4).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..m2.source import SOURCES, sha256_file
from ..split import answer_for_row

INDEX_VERSION = "wanxiang-retrieval-index-1"
ENTRIES_FILE = "entries.jsonl"
MANIFEST_FILE = "manifest.json"
EXPECTED_FILES = (ENTRIES_FILE, MANIFEST_FILE)


class IndexError(ValueError):
    """Index build/verify failure."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))


def build_index(source_json: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Build the retrieval index from the pinned source. Idempotent.

    Every row with a usable (word, answer) pair is indexed, including words in
    the eval-3 holdout. Duplicated words keep the first row deterministically.
    """
    src = Path(source_json)
    out = Path(out_dir)
    if not src.is_file():
        raise IndexError(f"source JSON not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise IndexError("source JSON must be an array")

    seen: set[str] = set()
    entries: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        word = str(row.get("word", "")).strip()
        if not word or word in seen:
            continue
        answer = answer_for_row(row)
        if not answer:
            continue
        seen.add(word)
        entries.append({
            "word": word,
            "answer": answer,
            "pinyin": str(row.get("pinyin", "") or ""),
        })
    if not entries:
        raise IndexError("no usable entries in source")

    # stable order by word so the index is reproducible regardless of source order
    entries.sort(key=lambda e: e["word"])
    entries_path = out / ENTRIES_FILE
    entries_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n"
                for e in entries),
        encoding="utf-8",
    )

    src_sha = sha256_file(src)
    pinned = SOURCES.get("idiom", {})
    doc: dict[str, Any] = {
        "version": INDEX_VERSION,
        "source": {
            "path": str(src),
            "sha256": src_sha,
            "expected_sha256": pinned.get("sha256", ""),
            "license": pinned.get("license", "unknown"),
        },
        "entries": len(entries),
        "answer_source": "autotrain.split.answer_for_row (shared with SFT + eval-3)",
        "holdout_excluded": False,
        "note": "Lookup tool: whole dictionary indexed by design, including "
                "eval-3 holdout words. This is NOT a generalisation claim.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    entries_sha = sha256_file(entries_path)
    doc["files"] = {
        ENTRIES_FILE: {"sha256": entries_sha,
                       "bytes": entries_path.stat().st_size,
                       "rows": len(entries)},
    }
    doc["index_id"] = hashlib.sha256(
        _canonical({k: v for k, v in doc.items() if k != "index_id"}).encode("utf-8")
    ).hexdigest()[:16]
    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify_index(index_dir: str | Path) -> dict[str, Any]:
    """Byte-check the index; fail on any tamper or truncation."""
    d = Path(index_dir)
    if not d.is_dir():
        raise IndexError(f"index directory not found: {d}")
    actual = sorted(p.name for p in d.iterdir() if p.is_file())
    if actual != sorted(EXPECTED_FILES):
        raise IndexError(f"index file set mismatch: {actual}")
    doc = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
    if doc.get("version") != INDEX_VERSION:
        raise IndexError("unsupported index version")
    for name, rec in doc["files"].items():
        f = d / name
        if sha256_file(f) != rec["sha256"] or f.stat().st_size != rec["bytes"]:
            raise IndexError(f"checksum/size mismatch: {name}")
    n = sum(1 for line in (d / ENTRIES_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip())
    if n != doc["files"][ENTRIES_FILE]["rows"]:
        raise IndexError("row count mismatch")
    sealed = {k: v for k, v in doc.items() if k != "index_id"}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]
    if expected != doc.get("index_id"):
        raise IndexError("index_id seal mismatch (manifest edited)")
    return doc


def read_entries(index_dir: str | Path) -> list[dict[str, str]]:
    return [json.loads(line)
            for line in (Path(index_dir) / ENTRIES_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()]
