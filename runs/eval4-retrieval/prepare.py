"""Build the sealed retrieval index for eval-4. Deterministic + verifiable.

    python runs/eval4-retrieval/prepare.py

Writes runs/eval4-retrieval/index/{entries.jsonl,manifest.json}.
No holdout exclusion: the index is a dictionary lookup tool. eval-4 reports
this openly and is INCOMPARABLE with the zero-shot generator eval-3.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.retrieval import build_index, verify_index  # noqa: E402
from autotrain.m2.source import sha256_file  # noqa: E402

SRC = ROOT / "runs/m2/sources/idiom.json"
INDEX = ROOT / "runs/eval4-retrieval/index"


def main() -> None:
    doc = build_index(SRC, INDEX)
    v = verify_index(INDEX)
    assert doc["index_id"] == v["index_id"]
    assert doc["source"]["sha256"] == sha256_file(SRC)
    assert doc["holdout_excluded"] is False
    print("INDEX_OK=" + json.dumps({
        "index_id": doc["index_id"],
        "entries": doc["entries"],
        "source_sha256": doc["source"]["sha256"][:16],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
