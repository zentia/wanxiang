"""HTTP contract for the retrieval-augmented chat endpoint."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from autotrain.config import ModelConfig
from autotrain.inference import IdentityTokenizer, InferenceEngine
from autotrain.inference.server import make_handler
from autotrain.retrieval import Retriever

VOCAB = 256
MAX_SEQ = 32


class _FakeHit:
    def __init__(self, answer: str):
        self.answer = answer
        self.method = "exact"

    def as_dict(self):
        return {"word": "白兔赤乌", "answer": self.answer, "score": 1.0,
                "method": "exact", "source": "retrieval"}


class _FakeRetriever:
    def best(self, content: str):
        if "白兔赤乌" in content:
            return _FakeHit("多借指时间")
        return None


class _FakeCfg:
    layers = 2
    dim = 16


class _FakeModel:
    cfg = _FakeCfg()

    def parameters(self):
        yield torch.zeros(1)


@pytest.fixture
def http_server():
    eng = InferenceEngine.__new__(InferenceEngine)
    eng.vocab_size = VOCAB
    eng.max_seq_len = MAX_SEQ
    eng.device = "cpu"
    eng.threads = 1
    eng.info = None
    eng.model = _FakeModel()

    handler = make_handler(
        eng, IdentityTokenizer(VOCAB), retriever=_FakeRetriever(),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    yield port
    server.shutdown()
    server.server_close()
    t.join(timeout=2)


def _post(port: int, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_health_reports_retrieval(http_server):
    port = http_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        health = json.loads(resp.read().decode("utf-8"))
    assert health["retrieval"] is True


def test_chat_exact_hit_answered_by_lookup(http_server):
    port = http_server
    status, body = _post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "请解释成语《白兔赤乌》的意思"}],
        "stream": False,
    })
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "多借指时间"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["retrieval"]["source"] == "retrieval"


def test_chat_exact_hit_stream_has_finish_reason(http_server):
    """SSE path must end with a finish_reason chunk then [DONE] (client contract)."""
    port = http_server
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": "请解释成语《白兔赤乌》的意思"}],
            "stream": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        raw = resp.read().decode("utf-8")
    chunks = [line[len("data: "):] for line in raw.splitlines()
              if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    payloads = [json.loads(c) for c in chunks if c != "[DONE]"]
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in payloads
    )
    assert content == "多借指时间"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["retrieval"]["source"] == "retrieval"


def test_chat_no_hit_without_generator_is_501(http_server):
    port = http_server
    status, body = _post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "随便聊点别的"}],
        "stream": False,
    })
    assert status == 501
    assert body["error"]["code"] == "chat_unavailable"
