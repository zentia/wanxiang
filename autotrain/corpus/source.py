"""Public corpus acquisition (network only needed for ``fetch``).

Sources are pinned by dataset/config/split/row range so a rebuild is
deterministic. The default source is Chinese Wikipedia via the HuggingFace
datasets-server ``/rows`` API, which returns plain text without requiring
``pyarrow``/``datasets``. Fetch results are sealed with sha256: any upstream
drift or truncation is detected before a corpus is built.

This module is deliberately domain-agnostic. No idiom / game assumptions.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://datasets-server.huggingface.co/rows"

# Predefined source profiles. ``n_rows`` is measured from ``offset``.
# Wikipedia zh snapshot is CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
SOURCES: dict[str, dict[str, Any]] = {
    "wiki_zh": {
        "api": API_BASE,
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "offset": 0,
        "n_rows": 400,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "description": "中文维基百科 20231101 快照正文（通用自然语言）",
    },
    # Extension block: rows 400..1199, disjoint from the pinned base corpus.
    # Used only to grow the TRAIN stream; the frozen eval holdout comes from
    # rows 0..399 and is never touched by this source.
    "wiki_zh_ext": {
        "api": API_BASE,
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "offset": 400,
        "n_rows": 800,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "description": "中文维基百科 20231101 快照正文（扩展训练区，与冻结评测集不重叠）",
    },
    # Second extension block: rows 1200..1999. Disjoint from both the frozen
    # base corpus (0..399, which owns the holdout) and wiki_zh_ext (400..1199).
    "wiki_zh_ext2": {
        "api": API_BASE,
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "offset": 1200,
        "n_rows": 800,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "description": "中文维基百科 20231101 快照正文（第二扩展训练区，仍与冻结评测集不重叠）",
    },
    # Third extension block: rows 2000..2799. Disjoint from the frozen base
    # corpus (0..399, which owns the holdout) and both prior extensions.
    "wiki_zh_ext3": {
        "api": API_BASE,
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "offset": 2000,
        "n_rows": 800,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "description": "中文维基百科 20231101 快照正文（第三扩展训练区，仍与冻结评测集不重叠）",
    },
}


class SourceError(RuntimeError):
    """Raised when a source spec is invalid or the download is untrustworthy."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def fetch(
    name: str = "wiki_zh",
    out_dir: str | Path = "runs/corpus/sources",
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Download a pinned source into ``<out_dir>/<name>.jsonl`` and seal it.

    One JSON object per line: ``{"id", "url", "title", "text"}``. Re-running
    with an identical cached file is a no-op (``status=cached``). A cached
    file whose checksum no longer matches the manifest is rejected rather
    than silently reused.
    """
    if name not in SOURCES:
        raise SourceError(f"unknown source {name!r}; available: {sorted(SOURCES)}")
    spec = SOURCES[name]
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    data_path = dest / f"{name}.jsonl"
    meta_path = dest / f"{name}.source.json"

    # Stable provenance fields that determine the exact byte stream we expect.
    pin = {
        "api": spec["api"],
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "offset": int(spec["offset"]),
        "n_rows": int(spec["n_rows"]),
    }

    cached_meta: dict[str, Any] | None = None
    if meta_path.is_file():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached_meta = None

    if (
        data_path.is_file()
        and not force
        and cached_meta is not None
        and cached_meta.get("pin") == pin
        and cached_meta.get("sha256") == sha256_file(data_path)
    ):
        return {**cached_meta, "status": "cached", "path": str(data_path)}

    # Fetch row pages. datasets-server caps length at 100 per request.
    records: list[dict[str, Any]] = []
    offset, remaining = pin["offset"], pin["n_rows"]
    while remaining > 0:
        length = min(100, remaining)
        params = urllib.parse.urlencode({
            "dataset": pin["dataset"],
            "config": pin["config"],
            "split": pin["split"],
            "offset": offset,
            "length": length,
        })
        url = f"{pin['api']}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "wanxiang-corpus/0.1"})
        # datasets-server occasionally returns transient 429/502/503 under a
        # burst of sequential page requests. Retry with exponential backoff;
        # the pinned (offset, length) is unchanged so data stays deterministic.
        payload = None
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                last_err = e
            except urllib.error.URLError as e:
                last_err = e
                if attempt == 4:
                    raise
            time.sleep(2 ** attempt)
        if payload is None:
            raise SourceError(f"failed to fetch offset {offset}: {last_err}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise SourceError(f"empty/invalid rows response at offset {offset}")
        for item in rows:
            row = item.get("row", {})
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            records.append({
                "id": row.get("id"),
                "url": row.get("url"),
                "title": row.get("title"),
                "text": text,
            })
        offset += length
        remaining -= length

    if len(records) < pin["n_rows"] // 2:
        raise SourceError(
            f"too few usable rows: {len(records)} of {pin['n_rows']} requested"
        )

    # Deterministic line order = API order; write with a trailing newline.
    with open(data_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    meta: dict[str, Any] = {
        "name": name,
        "pin": pin,
        "status": "downloaded",
        "path": str(data_path),
        "sha256": sha256_file(data_path),
        "bytes": data_path.stat().st_size,
        "rows": len(records),
        "license": spec["license"],
        "license_url": spec["license_url"],
        "description": spec["description"],
    }
    meta_path.write_text(_canonical(meta) + "\n", encoding="utf-8")
    return meta
