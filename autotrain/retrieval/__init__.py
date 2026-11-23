"""Deterministic idiom retrieval layer (M4).

成语释义 is a lookup task; a generator that never saw the holdout definition
cannot derive it from the glyphs. This package gives Wanxiang a guarded,
sealed retrieval tool that answers from the pinned dictionary and labels its
source honestly.

    from autotrain.retrieval import build_index, Retriever
    build_index("runs/m2/sources/idiom.json", "runs/eval4-retrieval/index")
    hit = Retriever.from_dir("runs/eval4-retrieval/index").best(
        "请解释成语《白兔赤乌》的意思"
    )
"""

from .index import (ENTRIES_FILE, INDEX_VERSION, MANIFEST_FILE, IndexError,
                    build_index, read_entries, verify_index)
from .retriever import RetrievalHit, Retriever, extract_word

__all__ = [
    "INDEX_VERSION", "ENTRIES_FILE", "MANIFEST_FILE",
    "IndexError", "build_index", "verify_index", "read_entries",
    "Retriever", "RetrievalHit", "extract_word",
]
