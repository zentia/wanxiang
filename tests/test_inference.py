"""Inference service: KV-cache correctness, determinism and HTTP contract.

The most important test here is ``test_cached_forward_matches_full_forward``:
if incremental decoding diverges from the training/eval forward pass, every
A/B report produced by the evaluator would become meaningless.
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
from autotrain.export.sampler import greedy_generate
from autotrain.inference import (
    GenerationConfig,
    IdentityTokenizer,
    InferenceEngine,
    TokenizerUnavailable,
)
from autotrain.inference.server import make_handler
from autotrain.trainer.model import WanxiangModel

VOCAB = 256
MAX_SEQ = 32


def _tiny_model(seed: int = 123) -> WanxiangModel:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return WanxiangModel(
            ModelConfig(layers=2, dim=64, heads=4, vocab_size=VOCAB),
            max_seq_len=MAX_SEQ,
        ).eval()


@pytest.fixture
def engine():
    original_threads = torch.get_num_threads()
    model = _tiny_model()
    eng = InferenceEngine(
        model, vocab_size=VOCAB, max_seq_len=MAX_SEQ, device="cpu", threads=1
    )
    yield eng, model
    torch.set_num_threads(original_threads)


# ---------------------------------------------------------------- cache math

def test_cached_forward_matches_full_forward(engine):
    """One-token-at-a-time decode must reproduce full-sequence logits."""
    _, model = engine
    idx = torch.tensor([[3, 7, 11, 19, 23, 31, 41, 53]], dtype=torch.long)
    with torch.no_grad():
        full, _ = model(idx)
        cache = None
        step_logits = []
        for t in range(idx.shape[1]):
            logits, cache = model.forward_cached(idx[:, t:t + 1], cache, t)
            step_logits.append(logits[0, 0])
    for t in range(idx.shape[1]):
        assert torch.allclose(full[0, t], step_logits[t], atol=1e-5, rtol=1e-5), (
            f"KV cache diverged at position {t}"
        )


def test_greedy_generate_matches_legacy_sampler(engine):
    """The service must agree with the existing read-only observation sampler."""
    eng, model = engine
    prompt = [3, 7, 11, 19, 23]
    produced = eng.generate(prompt, GenerationConfig(max_new_tokens=10)).token_ids
    with torch.no_grad():
        expected = greedy_generate(model, prompt, 10)
    assert produced == expected


def test_sampling_is_seeded_and_does_not_touch_global_rng(engine):
    eng, _ = engine
    prompt = [1, 2, 3]
    cfg = GenerationConfig(max_new_tokens=8, do_sample=True, temperature=0.8,
                           top_k=20, seed=7)
    before = torch.get_rng_state().clone()
    a = eng.generate(prompt, cfg).token_ids
    b = eng.generate(prompt, cfg).token_ids
    after = torch.get_rng_state()
    assert a == b
    assert torch.equal(before, after), "inference polluted the global torch RNG"
    assert all(0 <= t < VOCAB for t in a)


def test_context_cap_and_invalid_prompt(engine):
    eng, _ = engine
    # Only MAX_SEQ - len(prompt) tokens fit; finish_reason must say so.
    result = eng.generate(list(range(30)), GenerationConfig(max_new_tokens=10))
    assert result.completion_tokens == 2
    assert result.finish_reason == "context_full"
    assert result.truncated is True
    with pytest.raises(ValueError, match="max_seq_len"):
        eng.generate(list(range(33)), GenerationConfig(max_new_tokens=1))
    with pytest.raises(ValueError):
        eng.generate([], GenerationConfig(max_new_tokens=1))
    with pytest.raises(ValueError):
        eng.generate([VOCAB], GenerationConfig(max_new_tokens=1))


def test_stop_and_eos(engine):
    eng, _ = engine
    greedy = eng.generate([1, 2, 3], GenerationConfig(max_new_tokens=16)).token_ids
    stop_id = greedy[0]
    result = eng.generate([1, 2, 3], GenerationConfig(
        max_new_tokens=16, stop_token_ids=(stop_id,)))
    assert result.finish_reason == "stop"
    assert result.token_ids == [stop_id]


# ---------------------------------------------------------------- tokenizer

def test_identity_tokenizer_is_explicit_about_no_text():
    tok = IdentityTokenizer(VOCAB)
    assert tok.supports_text is False
    assert tok.decode([1, 2, 3]) == ""
    with pytest.raises(TokenizerUnavailable, match="no tokenizer"):
        tok.encode("hello")


# ---------------------------------------------------------------- HTTP

def _request(port: int, path: str, payload=None, api_key=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def test_http_endpoints(engine):
    eng, _ = engine
    handler = make_handler(eng, IdentityTokenizer(VOCAB), api_key="secret",
                           max_concurrency=1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        # health / models
        status, body = _request(port, "/health")
        assert status == 200
        assert json.loads(body)["status"] == "ok"
        status, body = _request(port, "/v1/models")
        assert status == 200
        assert json.loads(body)["data"][0]["supports_text"] is False

        # auth
        status, _ = _request(port, "/v1/token-completions", {"prompt_ids": [1, 2, 3]})
        assert status == 401

        # token completion (the only working generation endpoint on M1)
        status, body = _request(port, "/v1/token-completions",
                                {"prompt_ids": [1, 2, 3], "max_tokens": 5},
                                api_key="secret")
        assert status == 200
        doc = json.loads(body)
        assert len(doc["token_ids"]) == 5
        assert doc["text"] is None
        assert doc["usage"]["total_tokens"] == 8
        assert "no tokenizer" in doc["note"]

        # SSE streaming
        status, body = _request(port, "/v1/token-completions",
                                {"prompt_ids": [1, 2, 3], "max_tokens": 3,
                                 "stream": True}, api_key="secret")
        assert status == 200
        assert body.count("data: ") >= 5  # 3 tokens + final + [DONE]
        assert "data: [DONE]" in body

        # honesty about M1 limitations
        status, body = _request(port, "/v1/completions", {"prompt": "hello"},
                                api_key="secret")
        assert status == 400
        assert json.loads(body)["error"]["code"] == "tokenizer_unavailable"
        status, body = _request(port, "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "hi"}]},
                                api_key="secret")
        assert status == 501
        assert json.loads(body)["error"]["code"] == "chat_unavailable"

        # input validation
        status, _ = _request(port, "/v1/token-completions",
                             {"prompt_ids": [VOCAB]}, api_key="secret")
        assert status == 400
        status, _ = _request(port, "/nope")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
