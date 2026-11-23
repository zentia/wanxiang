"""SFT prototype: chat template, instruction corpus, /v1/chat/completions.

No network: all corpora are built from small local JSON sources.
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
from autotrain.inference import (CharTokenizer, ChatTemplate, ChatTemplateError,
                                  GenerationConfig, IdentityTokenizer,
                                  InferenceEngine)
from autotrain.inference.server import make_handler
from autotrain.sft.corpus import SFT_VERSION, SFTError, build, verify
from autotrain.trainer.model import WanxiangModel

# Every character used by the template and the question templates must be in
# the tiny test vocab (mirrors the real 8192 char vocab).
TINY_TEXT = (
    "用户问 助手答 系统说 \n请解释成语《画蛇添足》的意思是什么含义"
    "请说出比喻指形容犹原不一人之。，《》？ "
) * 30

ROWS = [
    {"word": "画蛇添足",
     "explanation": "比喻做了多余的事，反而不恰当。",
     "derivation": "语出《战国策》。"},
    {"word": "亡羊补牢",
     "explanation": "比喻出了问题以后想办法补救。",
     "derivation": "语出《战国策·楚策》。"},
]


def _tokenizer(tmp_path, vocab_size: int = 64) -> CharTokenizer:
    tok = CharTokenizer.build_from_text(TINY_TEXT, vocab_size=vocab_size)
    p = tmp_path / "tok.json"
    tok.save(p)
    return CharTokenizer.from_file(p)


def _source(tmp_path, repeats: int = 40) -> str:
    p = tmp_path / "src.json"
    p.write_text(json.dumps(ROWS * repeats, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _engine(vocab_size: int = 64) -> InferenceEngine:
    with torch.random.fork_rng():
        torch.manual_seed(11)
        model = WanxiangModel(
            ModelConfig(layers=2, dim=64, heads=4, vocab_size=vocab_size),
            max_seq_len=32,
        ).eval()
    return InferenceEngine(model, vocab_size=vocab_size, max_seq_len=32,
                           device="cpu", threads=1)


# ---------------------------------------------------------------- template

def test_chat_template_is_sealed_and_validates_chars(tmp_path):
    tok = _tokenizer(tmp_path)
    tmpl = ChatTemplate.default()
    tmpl.validate_chars(tok)  # all markers exist in this vocab
    p = tmp_path / "chat.json"
    tmpl.save(p)
    assert ChatTemplate.from_file(p).template_hash == tmpl.template_hash

    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["spec"]["role_prefix"]["user"] = "改"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ChatTemplateError, match="seal mismatch"):
        ChatTemplate.from_file(p)

    # a vocab without the markers must be refused, not silently mapped to <unk>
    poor = CharTokenizer.build_from_text("甲乙丙丁" * 30, vocab_size=64)
    with pytest.raises(ChatTemplateError, match="missing from vocab"):
        ChatTemplate.default().validate_chars(poor)


def test_encode_messages_and_extract_reply(tmp_path):
    tok = _tokenizer(tmp_path)
    tmpl = ChatTemplate.default()
    ids = tmpl.encode_messages([
        {"role": "system", "content": "简明回答"},
        {"role": "user", "content": "请解释成语《画蛇添足》的意思"},
    ], tok)
    assert ids[0] == 2  # <bos>
    assert ids[-len(tok.encode(tmpl.generation_prefix)):] == tok.encode(tmpl.generation_prefix)
    assert "用户问" in tok.decode(ids)

    assert tmpl.extract_reply("指人。\n用户问 下一题") == "指人。"
    assert tmpl.extract_reply("用户问 助手答 跑偏了") == ""
    with pytest.raises(ChatTemplateError):
        tmpl.encode_messages([{"role": "bad", "content": "x"}], tok)


def test_encode_messages_fitted_keeps_recent_turns_within_budget(tmp_path):
    """Long chat histories must be truncated, not overflow the model context."""
    tok = _tokenizer(tmp_path)
    tmpl = ChatTemplate.default()
    gen = tok.encode(tmpl.generation_prefix)
    long = "请解释成语" * 50
    messages = [
        {"role": "system", "content": long},
        {"role": "user", "content": long},
        {"role": "assistant", "content": long},
        {"role": "user", "content": "画蛇添足"},
    ]

    budget = 24
    ids = tmpl.encode_messages_fitted(messages, tok, max_prompt_tokens=budget)
    assert ids[0] == 2  # <bos> always kept
    assert ids[-len(gen):] == gen  # generation prefix always kept
    assert len(ids) <= budget
    # The most recent user question survives; the ancient system turn is dropped.
    decoded = tok.decode(ids)
    assert "用户问" in decoded
    assert "画蛇添足" in decoded
    assert "系统说" not in decoded

    # Even a single oversized message must produce a valid, bounded prompt.
    one = tmpl.encode_messages_fitted(
        [{"role": "user", "content": long}], tok, max_prompt_tokens=12
    )
    assert one[0] == 2
    assert one[-len(gen):] == gen
    assert len(one) <= 12


# ---------------------------------------------------------------- corpus

def test_sft_corpus_build_verify_and_tamper(tmp_path):
    tok = _tokenizer(tmp_path)
    tok_path = tmp_path / "tok.json"
    src = _source(tmp_path)
    out = tmp_path / "corpus"
    doc = build(src, tok_path, out)
    assert doc["version"] == SFT_VERSION
    assert doc["files"]["sft_tokens.bin"]["tokens"] > 1024
    assert verify(out)["corpus_id"] == doc["corpus_id"]

    out2 = tmp_path / "corpus2"
    assert build(src, tok_path, out2)["corpus_id"] == doc["corpus_id"]

    tokens = out / "sft_tokens.bin"
    raw = bytearray(tokens.read_bytes())
    raw[50] ^= 0xFF
    tokens.write_bytes(raw)
    with pytest.raises(SFTError, match="checksum mismatch"):
        verify(out)

    raw[50] ^= 0xFF
    tokens.write_bytes(raw)
    m = out / "manifest.json"
    mdoc = json.loads(m.read_text(encoding="utf-8"))
    mdoc["corpus_id"] = "0000000000000000"
    m.write_text(json.dumps(mdoc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SFTError, match="seal mismatch"):
        verify(out)


def test_sft_answer_side_labels(tmp_path):
    """v2 corpus emits a uint8 label stream parallel to tokens."""
    import numpy as _np
    tok = _tokenizer(tmp_path)
    tok_path = tmp_path / "tok.json"
    src = _source(tmp_path)
    out = tmp_path / "corpus"
    doc = build(src, tok_path, out)
    assert doc["extraction"]["answer_side_loss"] is True
    tokens = _np.fromfile(out / "sft_tokens.bin", dtype="<u2")
    labels = _np.fromfile(out / "sft_labels.bin", dtype=_np.uint8)
    assert labels.shape == tokens.shape
    assert labels.min() == 0 and labels.max() == 1
    # first token of every sample is BOS (label 0); answers contribute labels
    assert labels[0] == 0
    assert 0 < labels.sum() < labels.size   # both prompt and answer present


def test_sft_exclude_words(tmp_path):
    """Holdout words never enter the v2 SFT stream."""
    tok = _tokenizer(tmp_path)
    tok_path = tmp_path / "tok.json"
    src = _source(tmp_path)
    out = tmp_path / "corpus_ex"
    doc = build(src, tok_path, out, exclude_words={"画蛇添足"})
    assert "画蛇添足" in doc["extraction"]["excluded_words"]
    assert doc["extraction"]["entries_used"] < build(src, tok_path, tmp_path / "full")["extraction"]["entries_used"]


# ---------------------------------------------------------------- HTTP chat

def _request(port: int, path: str, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def test_chat_endpoint_enabled_only_with_template(tmp_path):
    original_threads = torch.get_num_threads()
    tok = _tokenizer(tmp_path)
    tmpl = ChatTemplate.default()
    tmpl.save(tmp_path / "chat.json")
    engine = _engine(vocab_size=64)
    handler = make_handler(engine, tok, chat_template=tmpl)
    plain = make_handler(engine, IdentityTokenizer(64))

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        status, body = _request(port, "/health")
        assert status == 200
        assert json.loads(body)["chat_completions"] is True

        q = {"messages": [{"role": "user", "content": "请解释成语《画蛇添足》的意思"}],
             "max_tokens": 8}
        status, body = _request(port, "/v1/chat/completions", q)
        assert status == 200
        doc = json.loads(body)
        assert doc["object"] == "chat.completion"
        assert doc["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(doc["choices"][0]["message"]["content"], str)
        assert doc["template_hash"] == tmpl.template_hash
        assert "raw_generated_text" in doc

        status, body = _request(port, "/v1/chat/completions",
                                {**q, "stream": True})
        assert status == 200 and "data: [DONE]" in body

        # A long, multi-turn history (as real chat clients resend) must not
        # overflow max_seq_len and must still stream to a clean [DONE].
        long = "请解释成语" * 200
        status, body = _request(port, "/v1/chat/completions", {
            "messages": [
                {"role": "system", "content": long},
                {"role": "user", "content": long},
                {"role": "assistant", "content": long},
                {"role": "user", "content": "画蛇添足"},
            ],
            "stream": True, "max_tokens": 8,
        })
        assert status == 200
        assert "data: [DONE]" in body
        # Terminal finish_reason must be an OpenAI-standard value; internal
        # eos/context_full tokens must never leak to the client.
        assert '"finish_reason": "eos"' not in body
        assert '"finish_reason": "context_full"' not in body
        terminal = [ln for ln in body.splitlines()
                    if '"finish_reason": "' in ln and '"finish_reason": null' not in ln]
        assert terminal, "no terminal chunk"
        assert any(f'"finish_reason": "{v}"' in terminal[-1]
                   for v in ("stop", "length"))

        status, body = _request(port, "/v1/chat/completions",
                                {"messages": [{"role": "bad", "content": "x"}]})
        assert status == 400
        assert json.loads(body)["error"]["code"] == "invalid_messages"
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)

    # Without a chat template the endpoint stays honestly unavailable.
    server2 = ThreadingHTTPServer(("127.0.0.1", 0), plain)
    server2.daemon_threads = True
    port2 = server2.server_address[1]
    t2 = threading.Thread(target=server2.serve_forever,
                          kwargs={"poll_interval": 0.05}, daemon=True)
    t2.start()
    try:
        status, body = _request(port2, "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "x"}]})
        assert status == 501
        assert json.loads(body)["error"]["code"] == "chat_unavailable"
    finally:
        server2.shutdown()
        server2.server_close()
        t2.join(timeout=5)
        torch.set_num_threads(original_threads)
