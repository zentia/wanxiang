"""Guarded idiom retriever: exact lookup + bigram fuzzy recall.

Design rules (kept deliberately simple and auditable):
  * Only text inside ``《》`` is treated as a candidate idiom word. This avoids
    retrieving on the question boilerplate (请解释成语...的意思).
  * Exact word match wins with score 1.0.
  * Otherwise recall by character-bigram overlap (Jaccard), with a hard floor
    so a wrong answer is never returned with high confidence.
  * Every result carries ``source=retrieval`` so the service can label it.

This is a deterministic lookup tool, not a learned ranker. It does not pretend
to generalise: if the word is absent from the index, confidence is low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .index import read_entries, verify_index

_WORD_RE = re.compile(r"《([^》]+)》")


@dataclass(frozen=True)
class RetrievalHit:
    word: str
    answer: str
    pinyin: str
    score: float
    method: str  # exact | bigram | none

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "answer": self.answer,
            "pinyin": self.pinyin,
            "score": round(self.score, 4),
            "method": self.method,
            "source": "retrieval",
        }


def _bigrams(word: str) -> set[str]:
    if len(word) < 2:
        return {word} if word else set()
    return {word[i:i + 2] for i in range(len(word) - 1)}


def extract_word(text: str) -> str | None:
    """Return the first 《...》 content, stripped."""
    m = _WORD_RE.search(text or "")
    if not m:
        return None
    w = m.group(1).strip()
    return w or None


class Retriever:
    def __init__(self, entries: list[dict[str, str]], *, min_score: float = 0.30):
        self._entries = entries
        self._by_word = {e["word"]: e for e in entries}
        self._grams: dict[str, set[str]] = {
            e["word"]: _bigrams(e["word"]) for e in entries
        }
        self.min_score = min_score

    @classmethod
    def from_dir(cls, index_dir: str | Path, *, min_score: float = 0.30) -> "Retriever":
        verify_index(index_dir)
        return cls(read_entries(index_dir), min_score=min_score)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        word = extract_word(query)
        if not word:
            return []
        exact = self._by_word.get(word)
        if exact:
            return [RetrievalHit(word=exact["word"], answer=exact["answer"],
                                 pinyin=exact["pinyin"], score=1.0, method="exact")]

        qg = _bigrams(word)
        if not qg:
            return []
        scored: list[RetrievalHit] = []
        for e in self._entries:
            g = self._grams[e["word"]]
            if not g:
                continue
            inter = len(qg & g)
            if not inter:
                continue
            jacc = inter / len(qg | g)
            if jacc >= self.min_score:
                scored.append(RetrievalHit(word=e["word"], answer=e["answer"],
                                           pinyin=e["pinyin"], score=jacc,
                                           method="bigram"))
        scored.sort(key=lambda h: (-h.score, h.word))
        return scored[:max(1, top_k)]

    def best(self, query: str) -> RetrievalHit | None:
        hits = self.retrieve(query, top_k=1)
        return hits[0] if hits else None
