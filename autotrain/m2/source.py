"""Public corpus acquisition (network only needed for ``fetch``).

Sources are pinned by URL + sha256 so a rebuild is deterministic and any
silent upstream change is detected. The corpus itself is never written into
``evals/``.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

# chinese-xinhua: MIT licensed Chinese idiom/dictionary dataset.
SOURCES: dict[str, dict[str, str]] = {
    "idiom": {
        "url": "https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/idiom.json",
        "repo": "https://github.com/pwxcoo/chinese-xinhua",
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/LICENSE",
        "sha256": "1d4b4f454ce1c416d6a1ab2369d6e66c0ff99e04390172eef70790499e21ce19",
        "description": "中文成语词典：30895 条，含典故/例句/释义等中文自然语言",
    },
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str = "idiom", out_dir: str | Path = "runs/m2/sources",
          *, force: bool = False) -> dict[str, Any]:
    """Download a pinned source; skip when the cached file already matches."""
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; available: {sorted(SOURCES)}")
    spec = SOURCES[name]
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.json"

    if path.is_file() and not force and sha256_file(path) == spec["sha256"]:
        status = "cached"
    else:
        req = urllib.request.Request(spec["url"],
                                     headers={"User-Agent": "wanxiang-m2/0.1"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        status = "downloaded"

    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"source checksum mismatch: expected {spec['sha256']}, got {actual}. "
            "Upstream changed; pin a new sha256 before rebuilding."
        )
    return {
        "name": name,
        "status": status,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "url": spec["url"],
        "repo": spec["repo"],
        "license": spec["license"],
        "license_url": spec["license_url"],
        "description": spec["description"],
    }
