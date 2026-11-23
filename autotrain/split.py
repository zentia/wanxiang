"""Deterministic word-level holdout split + answer cleaning.

The eval-2 instrument leaked: its holdout was defined by a token *offset*
in the SFT stream, but the long run wrapped around the whole training region
~19 times, so the "holdout" samples were memorised. This module fixes that at
the source: holdout is defined by **idiom word**, and every sample of a
holdout word is excluded from BOTH the M2 pretrain stream and the SFT stream.

``clean_answer`` is shared by training and evaluation so the reference the
model is scored against is byte-identical to the answer it was trained on.
It deliberately strips dictionary artefacts (互见条目 / 例句 / 残引号) that
made exact_match structurally impossible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Frozen constants -- changing them invalidates every eval-3 report.
HOLDOUT_SEED = 20261123
HOLDOUT_COUNT = 300          # 150 scored + spare; deterministic prefix
MAX_ANSWER_CHARS = 24        # fits in GEN_TOKENS=32 at char level

_MUTUAL_RE = re.compile(r"^(同|见|犹|即|亦作|又作|通)[^。；，]{0,6}$")
_SENT_SPLIT_RE = re.compile(r"[。；;]")
_DEFINE_HINT = ("指", "形容", "比喻", "表示", "意思是", "用来", "谓")


class SplitError(ValueError):
    """Raised when a holdout split cannot be built."""


def holdout_words(
    rows: list[dict[str, Any]],
    seed: int = HOLDOUT_SEED,
    count: int = HOLDOUT_COUNT,
) -> list[str]:
    """Pick ``count`` idiom words deterministically, sorted by sha256(seed:word).

    The selection depends only on the words themselves plus the seed, so a
    rebuild with the same pinned source yields the identical split. Changing
    the seed is a deliberate instrument change and must re-seal eval-3.
    """
    words = sorted({str(r.get("word", "")).strip() for r in rows
                    if isinstance(r, dict) and str(r.get("word", "")).strip()})
    if len(words) < count:
        raise SplitError(f"only {len(words)} distinct words, need {count}")
    keyed = [(hashlib.sha256(f"{seed}:{w}".encode("utf-8")).hexdigest(), w)
             for w in words]
    keyed.sort()
    return [w for _, w in keyed[:count]]


def split_rows(
    rows: list[dict[str, Any]],
    holdout: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition source rows into (train, holdout) by word membership."""
    train, held = [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        (held if str(row.get("word", "")).strip() in holdout else train).append(row)
    return train, held


def clean_answer(raw: str) -> str | None:
    """Turn a dictionary answer field into one short, clean definition.

    Rules (all deterministic, shared by training and eval-3):

    1. collapse to one line, drop control characters;
    2. cut at ``◇`` -- everything after it is an example/extension;
    3. drop stray Chinese quote marks left over from the source;
    4. split into sentences and pick the best one: prefer a sentence with a
       definition cue (指/形容/比喻/...), skip bare 互见条目 (同X、见X、犹X);
    5. truncate to ``MAX_ANSWER_CHARS`` characters.

    Returns ``None`` when nothing usable remains so callers can drop the row.
    """
    if not isinstance(raw, str):
        return None
    text = "".join(ch for ch in raw if ch.isprintable() and ch != "\n")
    text = " ".join(text.split())
    if not text:
        return None
    text = text.split("◇", 1)[0]
    for mark in ("“", "”", "\"", "'", "《", "》"):
        text = text.replace(mark, "")
    text = text.strip()
    if not text:
        return None

    sentences = [s.strip(" ，,") for s in _SENT_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if len(s) >= 3]
    if not sentences:
        return None

    def is_mutual(s: str) -> bool:
        return bool(_MUTUAL_RE.match(s))

    cued = [s for s in sentences
            if any(h in s for h in _DEFINE_HINT) and not is_mutual(s)]
    pool = cued or [s for s in sentences if not is_mutual(s)] or sentences
    best = pool[0]
    if len(best) > MAX_ANSWER_CHARS:
        best = best[:MAX_ANSWER_CHARS]
    return best or None


def answer_for_row(row: dict[str, Any]) -> str | None:
    """Extract and clean the answer from one dictionary row."""
    for key in ("explanation", "derivation", "example"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            cleaned = clean_answer(val)
            if cleaned:
                return cleaned
    return None
