"""Versioned chat template for the SFT prototype.

The M2 character vocab comes from a Chinese idiom dictionary: it contains
almost no ASCII letters and no full-width colon. So the template uses only
characters that are guaranteed to be in the frozen 8192 vocab (``用户问`` /
``助手答`` / ``系统说`` / ``\\n``). ``validate_chars`` enforces this instead of
silently letting role markers collapse into ``<unk>``.

The template is hashed and stored next to the SFT corpus, so a prompt format
change is detectable and reports stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .tokenizer import (BOS_ID, EOS_ID, CharTokenizer, CharTokenizerError,
                        Tokenizer, _canonical)

TEMPLATE_VERSION = "wanxiang-chat-1"

DEFAULT_SPEC: dict[str, Any] = {
    "name": TEMPLATE_VERSION,
    "role_prefix": {
        "system": "系统说 ",
        "user": "用户问 ",
        "assistant": "助手答 ",
    },
    "turn_sep": "\n",
    "generation_prefix": "助手答 ",
    "note": "Pure-Chinese markers because the M2 char vocab has no ASCII "
            "uppercase / full-width colon. Special tokens are ids, not text.",
}


class ChatTemplateError(ValueError):
    """Template file missing, malformed or tampered with."""


class ChatTemplate:
    def __init__(self, spec: dict[str, Any]):
        if spec.get("name") != TEMPLATE_VERSION:
            raise ChatTemplateError(f"unsupported template: {spec.get('name')}")
        roles = spec.get("role_prefix")
        if not isinstance(roles, dict) or not {"system", "user", "assistant"} <= set(roles):
            raise ChatTemplateError("role_prefix must define system/user/assistant")
        self.spec = dict(spec)
        self.name = spec["name"]
        self.role_prefix = dict(roles)
        self.turn_sep = spec["turn_sep"]
        self.generation_prefix = spec["generation_prefix"]
        self.template_hash = hashlib.sha256(
            _canonical(self.spec).encode("utf-8")
        ).hexdigest()

    @classmethod
    def default(cls) -> "ChatTemplate":
        return cls(DEFAULT_SPEC)

    # ---------------------------------------------------------------- serde
    def as_dict(self) -> dict[str, Any]:
        return {
            "version": TEMPLATE_VERSION,
            "spec": self.spec,
            "template_hash": self.template_hash,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = self.as_dict()
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
        return doc

    @classmethod
    def from_file(cls, path: str | Path) -> "ChatTemplate":
        p = Path(path)
        if not p.is_file():
            raise ChatTemplateError(f"chat template not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ChatTemplateError(f"invalid template JSON: {e}") from e
        tmpl = cls(doc["spec"])
        if doc.get("version") != TEMPLATE_VERSION or doc.get("template_hash") != tmpl.template_hash:
            raise ChatTemplateError("chat template seal mismatch")
        return tmpl

    # ---------------------------------------------------------------- checks
    def validate_chars(self, tokenizer: Tokenizer) -> None:
        """Fail loudly if any marker character is missing from the vocab."""
        if not getattr(tokenizer, "supports_text", False):
            raise ChatTemplateError("chat template requires a text tokenizer")
        needed = "".join(self.role_prefix.values()) + self.turn_sep + self.generation_prefix
        missing = sorted({ch for ch in needed
                          if ch not in getattr(tokenizer, "token_to_id", {})})
        if missing:
            raise ChatTemplateError(
                f"template characters missing from vocab: {missing!r}"
            )

    # ---------------------------------------------------------------- prompt
    def encode_messages(self, messages: list[dict[str, Any]],
                        tokenizer: CharTokenizer, *,
                        add_generation_prompt: bool = True) -> list[int]:
        """Turn OpenAI-style messages into a uint16 token-id prompt."""
        ids: list[int] = [BOS_ID]
        for msg in messages:
            if not isinstance(msg, dict):
                raise ChatTemplateError("each message must be an object")
            role = msg.get("role")
            content = msg.get("content", "")
            if role not in self.role_prefix:
                raise ChatTemplateError(f"unsupported role: {role!r}")
            if not isinstance(content, str):
                raise ChatTemplateError("message content must be a string")
            ids.extend(tokenizer.encode(
                self.role_prefix[role] + content + self.turn_sep
            ))
        if add_generation_prompt:
            ids.extend(tokenizer.encode(self.generation_prefix))
        return ids

    def encode_messages_fitted(
        self, messages: list[dict[str, Any]], tokenizer: CharTokenizer, *,
        max_prompt_tokens: int, add_generation_prompt: bool = True,
    ) -> list[int]:
        """Encode messages within a token budget, keeping the most recent turns.

        Real chat clients resend the full conversation (system + history). A
        small-context model must not 500 on that: drop the oldest turns until
        the encoded prompt fits, always preserving BOS and the generation
        prefix. Returns a prompt of at most ``max_prompt_tokens`` ids.
        """
        gen_ids = tokenizer.encode(self.generation_prefix) if add_generation_prompt else []
        fixed = 1 + len(gen_ids)  # BOS + generation prefix
        room = max(0, max_prompt_tokens - fixed)

        blocks: list[list[int]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ChatTemplateError("each message must be an object")
            role = msg.get("role")
            content = msg.get("content", "")
            if role not in self.role_prefix:
                raise ChatTemplateError(f"unsupported role: {role!r}")
            if not isinstance(content, str):
                raise ChatTemplateError("message content must be a string")
            blocks.append(tokenizer.encode(
                self.role_prefix[role] + content + self.turn_sep
            ))

        kept: list[list[int]] = []
        used = 0
        for block in reversed(blocks):
            if used + len(block) <= room:
                kept.append(block)
                used += len(block)
            else:
                break  # oldest messages are dropped; consecutive recent turns kept
        kept.reverse()

        ids = [BOS_ID]
        for block in kept:
            ids.extend(block)
        ids.extend(gen_ids)
        return ids

    def extract_reply(self, text: str) -> str:
        """Cut the assistant reply out of decoded generated text."""
        reply = text.split(self.turn_sep, 1)[0]
        for prefix in self.role_prefix.values():
            reply = reply.split(prefix, 1)[0]
        return reply.strip()

    def training_text(self, question: str, answer: str) -> str:
        """One SFT sample as flat text (special tokens added by the builder)."""
        return (self.role_prefix["user"] + question + self.turn_sep
                + self.role_prefix["assistant"] + answer)

    @property
    def eos_token_id(self) -> int:
        return EOS_ID
