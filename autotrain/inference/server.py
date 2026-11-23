"""Zero-dependency HTTP inference server (standard library only).

OpenAI-shaped on purpose. Three modes, all honest about what the model is:

* default M1   : ``/v1/token-completions`` only; text 400, chat 501
* ``--tokenizer`` (M2) : text completions return Chinese characters
* ``--tokenizer`` + ``--chat-template`` (SFT) : ``/v1/chat/completions`` works

It never fabricates text from token ids. See INFERENCE.md / M2.md / SFT.md.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

from .chat import ChatTemplate
from .engine import GenerationConfig, InferenceEngine
from .tokenizer import (IdentityTokenizer, Tokenizer, TokenizerUnavailable,
                        validate_token_ids)

MAX_BODY_BYTES = 64 * 1024


class HttpError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None,
                 error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type


def _error_payload(exc: HttpError) -> dict[str, Any]:
    err: dict[str, Any] = {"message": exc.message, "type": exc.error_type}
    if exc.code:
        err["code"] = exc.code
    return {"error": err}


# Engine-internal reasons -> OpenAI-standard values. OpenAI clients only
# recognise a small enum (stop / length / tool_calls / content_filter ...).
# Emitting "eos" or "context_full" makes them treat the stream as having no
# finish_reason at all (e.g. "Stream ended without finish_reason").
_FINISH_REASON_MAP = {
    "eos": "stop",
    "stop": "stop",
    "length": "length",
    "context_full": "length",
}


def _openai_finish_reason(reason: str) -> str:
    return _FINISH_REASON_MAP.get(reason, "stop")


def build_config(body: dict[str, Any], engine: InferenceEngine) -> tuple[GenerationConfig, bool]:
    """Validate request fields -> (GenerationConfig, stream)."""
    def _int(key: str, default: int) -> int:
        v = body.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int):
            raise HttpError(400, f"{key} must be an integer")
        return v

    def _float(key: str, default: float) -> float:
        v = body.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise HttpError(400, f"{key} must be a number")
        return float(v)

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise HttpError(400, "stream must be a boolean")

    stops = body.get("stop_token_ids", [])
    if not isinstance(stops, list):
        raise HttpError(400, "stop_token_ids must be a list of integers")
    if stops:
        try:
            validate_token_ids(stops, engine.vocab_size)
        except ValueError as e:
            raise HttpError(400, str(e)) from e

    eos = body.get("eos_token_id", None)
    if eos is not None and (isinstance(eos, bool) or not isinstance(eos, int)
                            or not 0 <= eos < engine.vocab_size):
        raise HttpError(400, "eos_token_id must be a valid token id or null")

    try:
        cfg = GenerationConfig(
            max_new_tokens=_int("max_tokens", 32),
            do_sample=bool(body.get("do_sample", False)),
            temperature=_float("temperature", 1.0),
            top_k=_int("top_k", 0),
            top_p=_float("top_p", 1.0),
            seed=_int("seed", 0),
            eos_token_id=eos,
            stop_token_ids=tuple(stops),
        )
        cfg.validate()
    except HttpError:
        raise
    except ValueError as e:
        raise HttpError(400, str(e)) from e
    return cfg, stream


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """Return the most recent user message text, or '' if none."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def make_handler(
    engine: InferenceEngine,
    tokenizer: Tokenizer,
    *,
    chat_template: Any | None = None,
    retriever: Any | None = None,
    api_key: str | None = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> type[BaseHTTPRequestHandler]:
    semaphore = threading.Semaphore(max(1, int(max_concurrency)))
    model_id = engine.model_id()
    chat_ready = chat_template is not None and tokenizer.supports_text
    # Retrieval can serve chat even without a chat template: a dictionary hit
    # needs no generation, so it is a legitimate fallback answering path.
    retrieval_ready = retriever is not None
    if chat_ready:
        note = ("SFT prototype: single-turn instruction format with a versioned "
                "Chinese chat template, trained 100 extra steps on QA pairs. "
                "Output is still not reliable prose.")
    elif tokenizer.supports_text:
        note = ("Character tokenizer: token ids map to characters. This is a "
                "general-purpose next-token LM, not an instruction-tuned chat "
                "model; output is raw continuation, not a chat answer.")
    else:
        note = (
            "M1 has no tokenizer. token ids are uint16 integers, not text; "
            "'text' is always null. Text and chat endpoints need --tokenizer / --chat-template."
        )

    class Handler(BaseHTTPRequestHandler):
        server_version = "WanxiangInference/0.1"
        protocol_version = "HTTP/1.0"
        _stream_started = False  # per-request override set by _sse_start

        # ------------------------------------------------------------ plumbing
        def log_message(self, fmt: str, *args: Any) -> None:
            if verbose:
                super().log_message(fmt, *args)

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_body(self) -> dict[str, Any]:
            length = self.headers.get("Content-Length")
            if length is None:
                raise HttpError(400, "Content-Length is required")
            try:
                n = int(length)
            except ValueError as e:
                raise HttpError(400, "invalid Content-Length") from e
            if n < 1 or n > MAX_BODY_BYTES:
                raise HttpError(413, f"request body must be 1..{MAX_BODY_BYTES} bytes")
            raw = self.rfile.read(n)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise HttpError(400, f"invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise HttpError(400, "request body must be a JSON object")
            return obj

        def _auth(self) -> None:
            if not api_key:
                return
            expected = f"Bearer {api_key}"
            if self.headers.get("Authorization") != expected:
                raise HttpError(401, "missing or invalid API key", "invalid_api_key",
                                "authentication_error")

        def _sse_start(self, completion_id: str) -> None:
            self._stream_started = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _sse(self, payload: dict[str, Any] | str) -> None:
            data = "[DONE]" if payload == "[DONE]" else json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            )
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        # ------------------------------------------------------------ GET
        def do_GET(self) -> None:
            try:
                path = urlsplit(self.path).path.rstrip("/") or "/"
                if path == "/health":
                    health = engine.health()
                    health["tokenizer"] = tokenizer.type
                    health["supports_text"] = tokenizer.supports_text
                    health["chat_completions"] = chat_ready
                    health["chat_template"] = (
                        chat_template.template_hash[:16] if chat_ready else None
                    )
                    health["retrieval"] = retrieval_ready
                    self._json(health)
                elif path == "/v1/models":
                    self._json({
                        "object": "list",
                        "data": [{
                            "id": model_id,
                            "object": "model",
                            "owned_by": "wanxiang",
                            "step": engine.info.step if engine.info else None,
                            "max_seq_len": engine.max_seq_len,
                            "vocab_size": engine.vocab_size,
                            "tokenizer": tokenizer.type,
                            "supports_text": tokenizer.supports_text,
                            "chat_completions": chat_ready,
                            "retrieval": retrieval_ready,
                        }],
                    })
                else:
                    raise HttpError(404, f"unknown path: {path}", "not_found")
            except HttpError as e:
                self._json(_error_payload(e), e.status)

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:
            try:
                self._auth()
                path = urlsplit(self.path).path.rstrip("/") or "/"
                if (path == "/v1/chat/completions"
                        and not chat_ready and not retrieval_ready):
                    raise HttpError(
                        501,
                        "chat/completions needs --chat-template (generation) "
                        "or --retrieval-index (lookup). Neither is enabled; "
                        "see SFT.md / EVAL4.md.",
                        "chat_unavailable",
                    )
                body = self._read_body()
                cfg, stream = build_config(body, engine)

                if path == "/v1/token-completions":
                    self._token_completions(body, cfg, stream)
                elif path == "/v1/completions":
                    self._completions(body, cfg, stream)
                elif path == "/v1/chat/completions":
                    self._chat_completions(body, cfg, stream)
                else:
                    raise HttpError(404, f"unknown path: {path}", "not_found")
            except HttpError as e:
                if self._stream_started:
                    self._stream_error(e.message)
                else:
                    self._json(_error_payload(e), e.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # keep the server alive on unexpected errors
                if verbose:
                    print(f"INFERENCE 500: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        print(f"INFERENCE 500 body: "
                              f"{json.dumps(body, ensure_ascii=False)[:2000]}", flush=True)
                    except Exception:
                        pass
                if self._stream_started:
                    # SSE headers (200) are already on the wire. Never send a
                    # second HTTP status; close the stream cleanly with an
                    # error chunk + [DONE] so clients don't report a truncated
                    # stream ("Stream ended without finish_reason").
                    self._stream_error(f"inference failed: {type(e).__name__}: {e}")
                else:
                    self._json(_error_payload(HttpError(
                        500, f"inference failed: {type(e).__name__}: {e}",
                        "internal_error", "server_error")), 500)

        def _stream_error(self, message: str) -> None:
            """Emit an in-stream error and terminate the SSE stream cleanly."""
            try:
                self._sse({
                    "id": "wxerror",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}],
                    "error": {"message": message, "type": "server_error"},
                })
                self._sse("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ------------------------------------------------------------ endpoints
        def _token_completions(self, body: dict[str, Any], cfg: GenerationConfig,
                               stream: bool) -> None:
            prompt_ids = body.get("prompt_ids")
            try:
                validate_token_ids(prompt_ids, engine.vocab_size)
            except (ValueError, TypeError) as e:
                raise HttpError(400, str(e)) from e

            completion_id = "wxcmpl-" + uuid.uuid4().hex[:16]
            created = int(time.time())

            if stream:
                self._sse_start(completion_id)
                idx = [0]

                def emit(token: int) -> None:
                    self._sse({
                        "id": completion_id,
                        "object": "token_completion.chunk",
                        "created": created,
                        "model": model_id,
                        "token_id": token,
                        "index": idx[0],
                    })
                    idx[0] += 1

                with semaphore:
                    result = engine.generate(prompt_ids, cfg, on_token=emit)
                self._sse({
                    "id": completion_id,
                    "object": "token_completion.chunk",
                    "finish_reason": _openai_finish_reason(result.finish_reason),
                    "usage": result.as_dict()["usage"],
                })
                self._sse("[DONE]")
                return

            with semaphore:
                result = engine.generate(prompt_ids, cfg)
            self._json({
                "id": completion_id,
                "object": "token_completion",
                "created": created,
                "model": model_id,
                "token_ids": result.token_ids,
                "text": None,
                "finish_reason": _openai_finish_reason(result.finish_reason),
                "usage": result.as_dict()["usage"],
                "cache_bytes": result.cache_bytes,
                "truncated": result.truncated,
                "note": note,
            })

        def _completions(self, body: dict[str, Any], cfg: GenerationConfig,
                         stream: bool) -> None:
            """OpenAI-shaped text completions. M1 rejects; M2 works unchanged."""
            if not tokenizer.supports_text:
                try:
                    tokenizer.encode(str(body.get("prompt", "")))
                except TokenizerUnavailable as e:
                    raise HttpError(400, str(e), "tokenizer_unavailable") from e

            prompt = body.get("prompt", "")
            if not isinstance(prompt, str):
                raise HttpError(400, "prompt must be a string")
            prompt_ids = tokenizer.encode(prompt)
            # Do not inject the tokenizer's eos id: this prototype's training
            # stream contains no BOS/EOS, so the model has never learned to emit
            # one. Callers can still pass eos_token_id explicitly.
            completion_id = "wxcmpl-" + uuid.uuid4().hex[:16]
            with semaphore:
                result = engine.generate(prompt_ids, cfg)
            text = tokenizer.decode(result.token_ids)
            if stream:
                self._sse_start(completion_id)
                self._sse({
                    "id": completion_id, "object": "text_completion",
                    "created": int(time.time()), "model": model_id,
                    "choices": [{"text": text, "index": 0,
                                 "finish_reason": _openai_finish_reason(result.finish_reason)}],
                    "usage": result.as_dict()["usage"],
                })
                self._sse("[DONE]")
                return
            self._json({
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"text": text, "index": 0,
                             "finish_reason": _openai_finish_reason(result.finish_reason)}],
                "usage": result.as_dict()["usage"],
                "token_ids": result.token_ids,
                "note": note,
            })

        def _chat_retrieval(self, answer: str, hit: Any, completion_id: str,
                            stream: bool) -> None:
            """Answer a chat turn from a sealed dictionary hit (no generation)."""
            created = int(time.time())
            meta = hit.as_dict()
            if stream:
                self._sse_start(completion_id)
                self._sse({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": answer},
                                 "finish_reason": None}],
                })
                self._sse({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}],
                    "retrieval": meta,
                })
                self._sse("[DONE]")
                return
            self._json({
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "retrieval": meta,
                "note": "Answered by sealed dictionary lookup, not the language "
                        "model. See EVAL4.md (incomparable with generation eval-3).",
            })

        def _chat_completions(self, body: dict[str, Any], cfg: GenerationConfig,
                              stream: bool) -> None:
            """OpenAI-shaped chat completion (retrieval hit or SFT generation)."""
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HttpError(400, "messages must be a non-empty list")

            # Guarded lookup: only an EXACT 《成语》 hit answers. Bigram/fuzzy
            # candidates are ignored here (they are for eval-4 diagnostics), so
            # the service never returns a confident wrong definition.
            if retrieval_ready:
                hit = retriever.best(_last_user_content(messages))
                if hit is not None and hit.method == "exact" and hit.answer:
                    self._chat_retrieval(
                        hit.answer, hit,
                        "wxchatcmpl-" + uuid.uuid4().hex[:16], stream,
                    )
                    return

            if not chat_ready:
                raise HttpError(
                    501,
                    "no exact retrieval hit and no --chat-template for generation",
                    "chat_unavailable",
                )
            # A real chat client resends the whole history. Keep the prompt
            # within the model context (reserving room for generation) so a
            # long conversation never 500s mid-stream.
            prompt_budget = max(1, engine.max_seq_len - cfg.max_new_tokens)
            try:
                prompt_ids = chat_template.encode_messages_fitted(
                    messages, tokenizer, max_prompt_tokens=prompt_budget
                )
            except Exception as e:
                raise HttpError(400, f"invalid messages: {e}", "invalid_messages") from e
            # The SFT stream ends every answer with <eos>, so the model has
            # actually seen this stop token (unlike the M2 pretrain stream).
            if cfg.eos_token_id is None:
                cfg = replace(cfg, eos_token_id=tokenizer.eos_token_id)

            completion_id = "wxchatcmpl-" + uuid.uuid4().hex[:16]
            created = int(time.time())

            if stream:
                self._sse_start(completion_id)
                stopped = [False]

                def emit(tid: int) -> None:
                    if stopped[0]:
                        return
                    ch = tokenizer.decode([tid])
                    if ch == chat_template.turn_sep:
                        stopped[0] = True
                        return
                    if not ch:
                        return
                    self._sse({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": ch},
                                     "finish_reason": None}],
                    })

                with semaphore:
                    result = engine.generate(prompt_ids, cfg, on_token=emit)
                self._sse({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": _openai_finish_reason(result.finish_reason)}],
                    "usage": result.as_dict()["usage"],
                })
                self._sse("[DONE]")
                return

            with semaphore:
                result = engine.generate(prompt_ids, cfg)
            raw_text = tokenizer.decode(result.token_ids)
            reply = chat_template.extract_reply(raw_text)
            self._json({
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": _openai_finish_reason(result.finish_reason),
                }],
                "usage": result.as_dict()["usage"],
                "token_ids": result.token_ids,
                "raw_generated_text": raw_text,
                "template_hash": chat_template.template_hash,
                "note": note,
            })

    return Handler


def serve(
    engine: InferenceEngine,
    tokenizer: Tokenizer,
    *,
    chat_template: ChatTemplate | None = None,
    retriever: Any | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> None:
    handler = make_handler(
        engine, tokenizer, chat_template=chat_template, retriever=retriever,
        api_key=api_key, max_concurrency=max_concurrency, verbose=verbose,
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    mode = "SFT/chat" if chat_template is not None and tokenizer.supports_text else (
        "M2/text" if tokenizer.supports_text else "M1/token-ids"
    )
    if retriever is not None:
        mode += "+retrieval"
    print(f"wanxiang inference: http://{host}:{port}  model={engine.model_id()} "
          f"device={engine.device} threads={engine.threads} mode={mode}", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: listening on a non-loopback address; set --api-key", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        print("wanxiang inference: stopped", flush=True)
