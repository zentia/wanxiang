"""M3 subword tokenizer: character-preserving BPE and cross-tokenizer A/B.

The BPE experiment changes ONLY the encoding granularity. These tests pin
that contract: vocab budget stays 8192, every corpus character remains
encodable, the file is checksum-sealed, and eval3 refuses to mix rulers
unless the change is explicitly declared via ``compare_tokenizers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrain.eval3 import compare_reports, compare_tokenizers
from autotrain.inference import (
    BPETokenizer,
    BPETokenizerError,
    CharTokenizer,
    CharTokenizerError,
    load_tokenizer,
)
from autotrain.m2.pipeline import build, verify

TEXT = (
    "画蛇添足比喻多此一举。亡羊补牢比喻出了问题以后想办法补救。"
    "守望相助形容彼此帮助。成语是汉语中的固定短语，意思常常不能只看字面。"
) * 12


def _base_report(**overrides):
    report = {
        "format_version": "wanxiang-eval3-report-1",
        "evalspec_hash": "a" * 64,
        "corpus_manifest_hash": "b" * 64,
        "device": "cpu",
        "chat_template_hash": "c" * 64,
        "qa_count": 150,
        "gen_tokens": 48,
        "tokenizer_type": "char",
        "tokenizer_hash": "d" * 64,
        "tokenizer_vocab_size": 8192,
        "bundle_step": 1350,
        "status": "OK",
        "metrics": {"composite": 0.0676},
    }
    report.update(overrides)
    return report


class TestBPETokenizer:
    def test_roundtrip_is_lossless_and_shorter_than_char(self):
        tok = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        ids = tok.encode(TEXT, add_bos=True, add_eos=True)
        assert tok.decode(ids) == TEXT
        assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id

        char = CharTokenizer.build_from_text(TEXT, vocab_size=256)
        assert len(ids) - 2 < len(char.encode(TEXT))  # subwords actually compress
        assert tok.vocab_size == char.vocab_size == 256
        assert tok.supports_text is True
        assert tok.eos_token_id == 3

    def test_every_corpus_char_is_encodable(self):
        tok = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        for ch in set(TEXT):
            if ch.isspace():
                continue
            assert tok.encode(ch)[0] != tok.unk_token_id, ch
        assert tok.encode("☃")[0] == tok.unk_token_id  # unseen char -> <unk>

    def test_pieces_are_bounded_and_deterministic(self):
        a = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        b = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        assert a.id_to_token == b.id_to_token and a.pieces == b.pieces
        assert a.pieces and all(2 <= len(p) <= 4 for p in a.pieces)
        assert len(set(a.pieces)) == len(a.pieces)
        assert a.n_merges == len(a.pieces)
        assert a.as_dict()["sha256"] == b.as_dict()["sha256"]

    def test_file_is_sealed_and_dispatched(self, tmp_path: Path):
        tok = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        p = tmp_path / "bpe.json"
        tok.save(p)

        loaded = load_tokenizer(p)
        assert isinstance(loaded, BPETokenizer)
        assert loaded.type == "bpe"
        assert loaded.encode(TEXT) == tok.encode(TEXT)
        assert loaded.as_dict()["sha256"] == tok.as_dict()["sha256"]

        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["pieces"][0] = doc["pieces"][0][::-1]
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(BPETokenizerError, match="checksum mismatch"):
            BPETokenizer.from_file(p)

    def test_load_tokenizer_rejects_unknown_type(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"type": "wordpiece", "vocab_size": 8}),
                     encoding="utf-8")
        with pytest.raises(CharTokenizerError, match="unsupported tokenizer type"):
            load_tokenizer(p)

    def test_char_and_bpe_hashes_differ_at_same_vocab(self):
        char = CharTokenizer.build_from_text(TEXT, vocab_size=256)
        bpe = BPETokenizer.build_from_text(TEXT, vocab_size=256)
        assert char.as_dict()["sha256"] != bpe.as_dict()["sha256"]
        assert char.vocab_size == bpe.vocab_size


class TestM2BPEPipeline:
    def test_build_bpe_corpus_is_sealed_and_verifiable(self, tmp_path: Path):
        src = tmp_path / "source.json"
        rows = [
            {
                "word": "画蛇添足",
                "explanation": "比喻多此一举。",
                "example": "这件事已经完成，再改就是画蛇添足了。",
            },
            {
                "word": "亡羊补牢",
                "explanation": "比喻出了问题以后想办法补救。",
                "example": "发现问题及时改正，亡羊补牢犹未为晚。",
            },
        ] * 120
        src.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

        out = tmp_path / "bpe_corpus"
        doc = build(src, out, vocab_size=256, tokenizer_kind="bpe")
        assert doc["tokenizer"]["type"] == "bpe"
        assert doc["tokenizer"]["vocab_size"] == 256
        assert doc["tokenizer"]["n_merges"] > 0
        assert doc["files"]["corpus_tokens.bin"]["tokens"] > 1024
        assert verify(out)["corpus_id"] == doc["corpus_id"]

        tok = load_tokenizer(out / "tokenizer.json")
        assert isinstance(tok, BPETokenizer)

    def test_unsupported_kind_fails_loudly(self, tmp_path: Path):
        src = tmp_path / "source.json"
        src.write_text(
            json.dumps([{"word": "甲乙", "explanation": "比喻测试固定短语。" * 200}],
                       ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="unsupported tokenizer_kind"):
            build(src, tmp_path / "bad", vocab_size=256, tokenizer_kind="word")


class TestCrossTokenizerCompare:
    def test_compatible_reports_give_a_verdict(self):
        before = _base_report()
        after = _base_report(
            tokenizer_type="bpe",
            tokenizer_hash="e" * 64,
            bundle_step=1400,
            metrics={"composite": 0.12},
        )
        result = compare_tokenizers(before, after)
        assert result["compatible"] is True
        assert result["cross_tokenizer"] is True
        assert result["verdict"] == "IMPROVED"
        assert result["delta_composite"] == pytest.approx(0.12 - 0.0676)

    def test_no_change_and_regression(self):
        before = _base_report()
        after = _base_report(tokenizer_hash="e" * 64)
        assert compare_tokenizers(before, after)["verdict"] == "NO_CHANGE"
        worse = _base_report(tokenizer_hash="e" * 64, metrics={"composite": 0.01})
        assert compare_tokenizers(before, worse)["verdict"] == "REGRESSED"

    def test_incompatible_when_protocol_differs(self):
        before = _base_report()
        for key, value in (
            ("device", "cuda"),
            ("gen_tokens", 64),
            ("qa_count", 100),
        ):
            after = _base_report(tokenizer_hash="e" * 64, **{key: value})
            result = compare_tokenizers(before, after)
            assert result["compatible"] is False
            assert result["verdict"] == "INCOMPARABLE"
            assert key in result["protocol_differences"]

    def test_incompatible_when_vocab_or_tokenizer_unchanged(self):
        before = _base_report()
        diff_vocab = _base_report(tokenizer_hash="e" * 64,
                                  tokenizer_vocab_size=4096)
        assert compare_tokenizers(before, diff_vocab)["verdict"] == "INCOMPARABLE"

        same_tok = _base_report(bundle_step=1400, metrics={"composite": 0.2})
        assert compare_tokenizers(before, same_tok)["verdict"] == "INCOMPARABLE"

    def test_strict_compare_still_rejects_tokenizer_change(self):
        before = _base_report()
        after = _base_report(tokenizer_hash="e" * 64,
                             metrics={"composite": 0.2})
        result = compare_reports(before, after)
        assert result["compatible"] is False
        assert result["verdict"] == "INCOMPARABLE"
