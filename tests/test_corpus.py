"""General-purpose corpus pipeline tests.

Pins domain-agnostic behaviour: document-level holdout that a training cursor
can never wrap into, char/BPE tokenizer sealed over full text, tamper-evident
manifest, and deterministic rebuilds. No idiom / game assumptions.
"""

from __future__ import annotations

import json

import pytest

from autotrain.corpus.pipeline import (
    CorpusError,
    build,
    clean_text,
    read_documents,
    split_documents,
    verify,
)
from autotrain.corpus.source import SOURCES


# Highly diverse text: the incrementing numbers make nearly every n-gram
# unique, so a BPE vocab of 512 cannot collapse the stream into a handful of
# tokens. This keeps both char and bpe token streams well above the floor.
DOCS = [
    {"text": "".join(
        f"第{d}-{k}节讨论样本{d}{k}的属性，取值{d}{k}影响结论{d}{k}，实验{d}{k}验证了模型{d}{k}的有效性。"
        for k in range(60)
    )}
    for d in range(8)
]


def _source(tmp_path):
    p = tmp_path / "source.jsonl"
    p.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DOCS),
                 encoding="utf-8")
    return p


class TestCleanText:
    def test_normalises_whitespace_and_control_chars(self):
        raw = "第一行\r\n第二行\tok\x00结束"
        out = clean_text(raw)
        assert "\x00" not in out
        assert "\t" not in out
        assert "第一行" in out and "第二行 ok" in out


def test_read_documents_and_split(tmp_path):
    src = _source(tmp_path)
    docs = read_documents(src)
    assert len(docs) == len(DOCS)
    train, holdout = split_documents(len(docs), holdout_fraction=0.2, seed=42)
    assert set(train).isdisjoint(holdout)
    assert set(train) | set(holdout) == set(range(len(docs)))
    # deterministic
    train2, holdout2 = split_documents(len(docs), holdout_fraction=0.2, seed=42)
    assert (train, holdout) == (train2, holdout2)


class TestBuildVerify:
    def test_build_creates_sealed_corpus(self, tmp_path):
        src = _source(tmp_path)
        out = tmp_path / "corpus"
        doc = build(src, out, vocab_size=512, holdout_fraction=0.2, seed=7)
        assert doc["tokenizer"]["type"] == "char"
        assert doc["files"]["train_tokens.bin"]["tokens"] > 1024
        assert doc["files"]["holdout_tokens.bin"]["tokens"] > 0
        assert len(doc["split"]["holdout_indices"]) >= 1
        assert verify(out)["corpus_id"] == doc["corpus_id"]

    def test_holdout_text_never_in_train_stream(self, tmp_path):
        """Holdout doc ids are excluded before concatenation."""
        src = _source(tmp_path)
        out = tmp_path / "corpus"
        doc = build(src, out, vocab_size=512, holdout_fraction=0.2, seed=7)
        holdout_ids = set(doc["split"]["holdout_indices"])
        train_ids = set(range(len(DOCS))) - holdout_ids
        holdout_text = "".join(DOCS[i]["text"] for i in holdout_ids)
        train_text = "".join(DOCS[i]["text"] for i in train_ids)
        assert holdout_text not in train_text

    def test_tamper_is_detected(self, tmp_path):
        src = _source(tmp_path)
        out = tmp_path / "corpus"
        build(src, out, vocab_size=512, holdout_fraction=0.2, seed=7)
        hp = out / "holdout_tokens.bin"
        raw = bytearray(hp.read_bytes())
        raw[0] ^= 0xFF
        hp.write_bytes(raw)
        with pytest.raises(CorpusError):
            verify(out)

    def test_bpe_kind(self, tmp_path):
        src = _source(tmp_path)
        out = tmp_path / "corpus"
        doc = build(src, out, vocab_size=512, tokenizer_kind="bpe",
                    holdout_fraction=0.2, seed=7)
        assert doc["tokenizer"]["type"] == "bpe"
        assert verify(out)["corpus_id"] == doc["corpus_id"]

    def test_too_short_source_rejected(self, tmp_path):
        src = tmp_path / "short.jsonl"
        src.write_text(json.dumps({"text": "太短"}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        with pytest.raises(CorpusError):
            build(src, tmp_path / "c", vocab_size=512)


class TestDisjointExpansionSource:
    """The train-only extension profile must stay disjoint from the base holdout."""

    def test_ext_profile_is_disjoint_row_range(self):
        base, ext, ext2, ext3 = (
            SOURCES["wiki_zh"], SOURCES["wiki_zh_ext"], SOURCES["wiki_zh_ext2"],
            SOURCES["wiki_zh_ext3"],
        )
        assert base["offset"] + base["n_rows"] <= ext["offset"], \
            "extension rows must not overlap the base corpus (and its holdout)"
        assert ext["offset"] + ext["n_rows"] <= ext2["offset"], \
            "second extension must not overlap the first extension"
        assert ext2["offset"] + ext2["n_rows"] <= ext3["offset"], \
            "third extension must not overlap the second extension"

    def test_split_is_deterministic_across_rebuilds(self):
        # Expanding train must not reshuffle which base docs are held out.
        a_train, a_hold = split_documents(400, holdout_fraction=0.05, seed=42)
        b_train, b_hold = split_documents(400, holdout_fraction=0.05, seed=42)
        assert a_train == b_train and a_hold == b_hold
