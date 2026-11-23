"""M2 minimal prototype: char tokenizer, versioned corpus, text endpoints.

No network here: ``fetch`` is pinned and tested separately; pipeline tests
use a small local JSON source in tmp_path.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from autotrain.config import ModelConfig
from autotrain.inference import CharTokenizer, CharTokenizerError, InferenceEngine
from autotrain.inference.server import make_handler
from autotrain.m2.pipeline import (CORPUS_VERSION, CorpusError, build,
                                   clean_text, extract_text, verify)
from autotrain.m2.source import SOURCES
from autotrain.trainer.model import WanxiangModel

SAMPLE_ROWS = [
    {
        "word": "画蛇添足",
        "derivation": "语出《战国策》，比喻多此一举。",
        "example": "这件事已经完成，再改就是画蛇添足了。",
        "explanation": "比喻做了多余的事，反而不恰当。",
        "pinyin": "hua she tian zu",
    },
    {
        "word": "亡羊补牢",
        "derivation": "语出《战国策·楚策》。",
        "example": "发现问题及时改正，亡羊补牢犹未为晚。",
        "explanation": "比喻出了问题以后想办法补救，可以防止继续受损失。",
        "pinyin": "wang yang bu lao",
    },
]


def _source(tmp_path, repeats: int = 40) -> str:
    rows = SAMPLE_ROWS * repeats
    p = tmp_path / "source.json"
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _tiny_engine(vocab_size: int = 64):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = WanxiangModel(
            ModelConfig(layers=2, dim=64, heads=4, vocab_size=vocab_size),
            max_seq_len=32,
        ).eval()
    return InferenceEngine(model, vocab_size=vocab_size, max_seq_len=32,
                           device="cpu", threads=1)


# ---------------------------------------------------------------- tokenizer

def test_char_tokenizer_roundtrip_and_determinism():
    text = "画蛇添足，亡羊补牢。\n中文测试 ABC 123。"
    a = CharTokenizer.build_from_text(text, vocab_size=64)
    b = CharTokenizer.build_from_text(text, vocab_size=64)
    assert a.id_to_token == b.id_to_token
    assert a.vocab_size == 64 and a.supports_text is True
    assert a.eos_token_id == 3 and a.bos_token_id == 2
    assert a.decode(a.encode(text, add_bos=True, add_eos=True)) == text
    assert a.encode("☃")[0] == a.unk_token_id  # unseen char -> <unk>
    assert a.used_chars < 64
    assert all(t.startswith("<unused_") for t in a.id_to_token[4 + a.used_chars:])


def test_char_tokenizer_file_is_sealed(tmp_path):
    tok = CharTokenizer.build_from_text("中文文本语料测试" * 20, vocab_size=64)
    p = tmp_path / "tokenizer.json"
    tok.save(p)
    assert CharTokenizer.from_file(p).id_to_token == tok.id_to_token
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["id_to_token"][10] = "改"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CharTokenizerError, match="checksum mismatch"):
        CharTokenizer.from_file(p)


# ---------------------------------------------------------------- cleaning

def test_extract_and_clean_are_deterministic(tmp_path):
    text, entries = extract_text(_source(tmp_path, 1))
    assert entries == 2
    assert "画蛇添足" in text
    assert "pinyin" not in text and "hua" not in text  # latin field excluded
    assert clean_text("甲\x00乙\r\n丙\t丁") == "甲乙\n丙 丁\n"


# ---------------------------------------------------------------- pipeline

def test_corpus_build_is_deterministic_and_sealed(tmp_path):
    src = _source(tmp_path)
    out = tmp_path / "corpus"
    doc = build(src, out, vocab_size=64)
    assert doc["version"] == CORPUS_VERSION
    assert doc["files"]["corpus_tokens.bin"]["tokens"] > 1024
    assert doc["tokenizer"]["vocab_size"] == 64
    assert verify(out)["corpus_id"] == doc["corpus_id"]

    # Rebuilding the same source must produce the same corpus_id.
    out2 = tmp_path / "corpus2"
    doc2 = build(src, out2, vocab_size=64)
    assert doc2["corpus_id"] == doc["corpus_id"]


def test_corpus_tamper_detection(tmp_path):
    out = tmp_path / "corpus"
    build(_source(tmp_path), out, vocab_size=64)

    tokens = out / "corpus_tokens.bin"
    raw = bytearray(tokens.read_bytes())
    raw[100] ^= 0xFF
    tokens.write_bytes(raw)
    with pytest.raises(CorpusError, match="checksum mismatch"):
        verify(out)

    # Restore, then tamper with the manifest seal.
    raw[100] ^= 0xFF
    tokens.write_bytes(raw)
    m = out / "manifest.json"
    doc = json.loads(m.read_text(encoding="utf-8"))
    doc["corpus_id"] = "deadbeefdeadbeef"
    m.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CorpusError, match="seal mismatch"):
        verify(out)


def test_source_is_pinned():
    spec = SOURCES["idiom"]
    assert spec["license"] == "MIT"
    assert len(spec["sha256"]) == 64
    assert spec["url"].startswith("https://")


# ---------------------------------------------------------------- HTTP text

def _request(port: int, path: str, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def test_text_completions_work_but_chat_is_still_refused(tmp_path):
    original_threads = torch.get_num_threads()
    text = "画蛇添足，亡羊补牢，中文语料测试。" * 30
    tok = CharTokenizer.build_from_text(text, vocab_size=64)
    tok.save(tmp_path / "tokenizer.json")
    tok = CharTokenizer.from_file(tmp_path / "tokenizer.json")
    engine = _tiny_engine(vocab_size=64)
    handler = make_handler(engine, tok)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        status, body = _request(port, "/health")
        health = json.loads(body)
        assert status == 200
        assert health["tokenizer"] == "char" and health["supports_text"] is True

        status, body = _request(port, "/v1/completions",
                                {"prompt": "画蛇添足", "max_tokens": 6})
        assert status == 200
        doc = json.loads(body)
        assert isinstance(doc["choices"][0]["text"], str)
        assert len(doc["token_ids"]) == 6
        # The service must honestly state this is not a chat model and that
        # its output is raw continuation. Assert the semantics, not the exact
        # wording: the note is stage-neutral and may mention no milestone name.
        note = doc["note"].lower()
        assert "not an instruction-tuned chat model" in note
        assert "continuation" in note or "not fluent" in note

        # Token-id endpoint still works alongside the text endpoint.
        status, body = _request(port, "/v1/token-completions",
                                {"prompt_ids": [5, 6, 7], "max_tokens": 3})
        assert status == 200

        # No SFT / chat template: still honestly rejected.
        status, body = _request(port, "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "你好"}]})
        assert status == 501
        assert json.loads(body)["error"]["code"] == "chat_unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        torch.set_num_threads(original_threads)
