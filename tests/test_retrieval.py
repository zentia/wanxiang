"""Tests for the sealed idiom retrieval layer (M4) + eval-4 instrument."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autotrain.eval4 import (COMPARABLE_WITH_EVAL3, Eval4Runner, compare_eval4,
                             eval4_hash, run_retrieval)
from autotrain.retrieval import (IndexError, Retriever, build_index,
                                 extract_word, read_entries, verify_index)


@pytest.fixture
def tiny_index(tmp_path):
    """Build a tiny sealed index from a synthetic source reusing the real schema."""
    src = tmp_path / "idiom.json"
    rows = [
        {"word": "白兔赤乌", "pinyin": "bai tu chi wu",
         "explanation": "多借指时间"},
        {"word": "暗无天日", "pinyin": "an wu tian ri",
         "explanation": "形容在反动势力统治下社会的黑暗"},
        {"word": "安危与共", "pinyin": "an wei yu gong",
         "explanation": "形容关系密切，利害相连"},
        {"word": "画蛇添足", "pinyin": "hua she tian zu",
         "explanation": "比喻做多余的事反而不恰当"},
    ]
    src.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "idx"
    doc = build_index(src, out)
    return out, doc


def test_extract_word():
    assert extract_word("请解释成语《白兔赤乌》的意思") == "白兔赤乌"
    assert extract_word("成语《画蛇添足》是什么意思") == "画蛇添足"
    assert extract_word("没有书名号的问题") is None
    assert extract_word("《》") is None


def test_build_and_verify_index(tiny_index):
    out, doc = tiny_index
    v = verify_index(out)
    assert v["index_id"] == doc["index_id"]
    assert v["entries"] == 4
    assert v["holdout_excluded"] is False
    entries = read_entries(out)
    assert [e["word"] for e in entries] == sorted(
        ["白兔赤乌", "暗无天日", "安危与共", "画蛇添足"]
    )
    by_word = {e["word"]: e["answer"] for e in entries}
    assert by_word["白兔赤乌"] == "多借指时间"


def test_index_tamper_detected(tiny_index):
    out, _doc = tiny_index
    entries_path = out / "entries.jsonl"
    original = entries_path.read_text(encoding="utf-8")
    entries_path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(IndexError):
        verify_index(out)
    entries_path.write_text(original, encoding="utf-8")


def test_exact_lookup_wins(tiny_index):
    out, _ = tiny_index
    r = Retriever.from_dir(out)
    hit = r.best("请解释成语《白兔赤乌》的意思")
    assert hit is not None
    assert hit.method == "exact"
    assert hit.answer == "多借指时间"
    assert hit.score == 1.0
    assert hit.as_dict()["source"] == "retrieval"


def test_no_match_returns_none(tiny_index):
    out, _ = tiny_index
    r = Retriever.from_dir(out)
    assert r.best("请解释《完全无关的词》") is None
    assert r.best("没有成语标记") is None
    assert r.retrieve("没有成语标记") == []


def test_fuzzy_recall_never_exact(tiny_index):
    out, _ = tiny_index
    r = Retriever.from_dir(out, min_score=0.2)
    # A near-miss word recalls bigram candidates but never claims exact.
    hits = r.retrieve("请解释《白兔赤鸟》", top_k=3)
    assert all(h.method == "bigram" for h in hits)
    assert all(h.score < 1.0 for h in hits)


def test_eval4_uses_frozen_corpus_and_is_incomparable(tiny_index):
    out, doc = tiny_index
    r = Retriever.from_dir(out)
    report = run_retrieval(r, index_hash=doc["index_id"])
    assert report["comparable_with_eval3"] is False
    assert COMPARABLE_WITH_EVAL3 is False
    assert report["qa_count"] == 150
    assert report["index_hash"] == doc["index_id"]
    assert report["evalspec_hash"] == eval4_hash(
        report["corpus_manifest_hash"], doc["index_id"]
    )
    # tiny index only has 4 words: exact hit rate must be low -> honest PARTIAL
    assert report["exact_hit_rate"] < 0.05
    assert report["checks"]["quality_passed"] is False


def test_compare_eval4_requires_same_corpus():
    gen = {
        "corpus_manifest_hash": "aaa",
        "metrics": {"composite": 0.08, "exact_match": 0.0},
    }
    ret = {
        "corpus_manifest_hash": "bbb",
        "metrics": {"composite": 0.99, "exact_match": 1.0},
    }
    assert compare_eval4(gen, ret)["verdict"] == "INCOMPARABLE"
    ret["corpus_manifest_hash"] = "aaa"
    cmp = compare_eval4(gen, ret)
    assert cmp["compatible"] is True
    assert cmp["verdict"] == "RETRIEVAL_SUPERIOR"
    assert cmp["delta"] == pytest.approx(0.91)
