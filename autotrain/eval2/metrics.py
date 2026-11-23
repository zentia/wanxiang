"""Objective, deterministic character-level answer-quality metrics.

No LLM-as-a-judge, no learned reward model, no network. Every function is
pure so the frozen spec can be tested independently of the model.
"""

from __future__ import annotations

from collections import Counter

from .spec import METRIC_WEIGHTS


def normalize(text: str) -> str:
    """Keep CJK / letters / digits, drop punctuation and whitespace."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def char_f1(pred: str, ref: str) -> float:
    """Bag-of-characters F1 (Chinese characters are the natural token unit)."""
    p, r = normalize(pred), normalize(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    cp, cr = Counter(p), Counter(r)
    overlap = sum(min(cp[ch], cr[ch]) for ch in cp.keys() & cr.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    """LCS-based F1 (order-sensitive complement to char_f1)."""
    p, r = normalize(pred), normalize(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    # O(len(p) * len(r)) with O(min) memory; answers here are short.
    if len(p) < len(r):
        p, r = r, p
    prev = [0] * (len(r) + 1)
    for ch in p:
        cur = [0]
        for j, rch in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if ch == rch else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    return max(0.0, 1.0 - _levenshtein(p, r) / max(len(p), len(r)))


def score_pair(pred: str, ref: str) -> dict[str, float]:
    parts = {
        "char_f1": char_f1(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "edit_similarity": edit_similarity(pred, ref),
        "exact_match": exact_match(pred, ref),
    }
    parts["composite"] = sum(METRIC_WEIGHTS[k] * v for k, v in parts.items())
    return parts


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {k: 0.0 for k in (*METRIC_WEIGHTS, "composite")}
    keys = (*METRIC_WEIGHTS, "composite")
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}


def empty_reply_rate(replies: list[str]) -> float:
    return sum(1 for r in replies if not normalize(r)) / max(1, len(replies))
