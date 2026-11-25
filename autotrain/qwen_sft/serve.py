"""Local OpenAI-shaped HTTP server for the 4-bit Qwen base + QLoRA adapter.

Zero extra dependencies: standard-library http.server plus the already
installed transformers / peft / torch. Single model instance, serialized
generation with a lock (HF generate is not thread-safe).

Decode controls default ON (repetition_penalty 1.2, no_repeat_ngram_size 3):
CODE_LM.md 9.7 measured them as NO_CHANGE on quality while removing all
greedy loops. Callers can override per request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

DEFAULT_DECODE = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3}


class HttpError(Exception):
    def __init__(self, code: int, message: str, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.code = code
        self.message = message
        self.error_type = error_type


def _payload(exc: HttpError) -> dict[str, Any]:
    return {"error": {"message": exc.message, "type": exc.error_type,
                      "code": exc.code}}


def run(
    base_dir: str,
    *,
    adapter_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_new_tokens: int = 256,
    api_key: str | None = None,
    load_in_4bit: bool = True,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    api_key = api_key or os.environ.get("WANXIANG_API_KEY")
    tok = AutoTokenizer.from_pretrained(base_dir)
    kw: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_dir, **kw)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lock = threading.Lock()
    model_id = f"{os.path.basename(base_dir)}{'+adapter' if adapter_dir else ''}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # keep stdout clean; health output is printed once below

        def _send(self, code: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _auth(self) -> None:
            if not api_key:
                return
            got = self.headers.get("Authorization", "")
            if got != f"Bearer {api_key}":
                raise HttpError(401, "invalid api key", "authentication_error")

        def do_GET(self) -> None:
            try:
                self._auth()
                if self.path != "/health":
                    raise HttpError(404, "not found")
                self._send(200, {
                    "status": "ok",
                    "model": model_id,
                    "base": os.path.basename(base_dir),
                    "adapter": bool(adapter_dir),
                    "quantization": "4bit-nf4" if load_in_4bit else "bf16",
                    "default_decode": DEFAULT_DECODE,
                    "endpoints": ["/health", "/v1/chat/completions"],
                })
            except HttpError as e:
                self._send(e.code, _payload(e))

        def do_POST(self) -> None:
            try:
                self._auth()
                if self.path != "/v1/chat/completions":
                    raise HttpError(404, "not found")
                length = int(self.headers.get("Content-Length", "0"))
                if not length:
                    raise HttpError(400, "empty body")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise HttpError(400, "messages must be a non-empty list")
                n = body.get("max_tokens", max_new_tokens)
                if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 2048:
                    raise HttpError(400, "max_tokens must be an integer in [1, 2048]")
                decode = dict(DEFAULT_DECODE)
                for key in ("repetition_penalty", "no_repeat_ngram_size"):
                    if key in body:
                        v = body[key]
                        if not isinstance(v, (int, float)) or isinstance(v, bool):
                            raise HttpError(400, f"{key} must be a number")
                        decode[key] = float(v) if key == "repetition_penalty" else int(v)
                do_sample = bool(body.get("do_sample", False))
                temperature = float(body.get("temperature", 1.0))
                if do_sample and temperature <= 0:
                    raise HttpError(400, "temperature must be > 0 when do_sample")

                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                x = tok(text, return_tensors="pt").to(model.device)
                with lock, torch.no_grad():
                    out = model.generate(
                        **x, max_new_tokens=n, do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        pad_token_id=pad_id, **decode,
                    )
                gen_ids = [int(t) for t in out[0][x["input_ids"].shape[1]:].tolist()]
                reply = tok.decode(gen_ids, skip_special_tokens=True)
                finished = "stop" if (gen_ids and tok.eos_token_id in gen_ids) else "length"
                self._send(200, {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": finished,
                    }],
                    "usage": {
                        "prompt_tokens": int(x["input_ids"].shape[1]),
                        "completion_tokens": len(gen_ids),
                        "total_tokens": int(x["input_ids"].shape[1]) + len(gen_ids),
                    },
                })
            except HttpError as e:
                self._send(e.code, _payload(e))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, _payload(HttpError(400, str(e))))

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({
        "health": f"http://{host}:{port}/health",
        "endpoint": f"http://{host}:{port}/v1/chat/completions",
        "model": model_id,
        "default_decode": DEFAULT_DECODE,
        "api_key_required": bool(api_key),
    }, ensure_ascii=False, indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
