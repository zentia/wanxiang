"""Build and verify the instruction-formatted SFT corpus.

Data source is the same pinned MIT idiom dictionary as M2, but turned into
single-turn ``用户问 / 助手答`` conversations. The chat template is pure
Chinese because the frozen 8192 char vocab has no ASCII uppercase / full-width
colon.

Boundary: this is TRAINING data. It is not an eval set, does not enter
``evals/``, and does not change ``evalspec_hash``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.chat import ChatTemplate, ChatTemplateError
from ..inference.tokenizer import (BOS_ID, EOS_ID, CharTokenizerError,
                                    _canonical, load_tokenizer)
from ..m2.source import SOURCES, sha256_file

SFT_VERSION = "wanxiang-m2-sft-2"
TOKENS_FILE = "sft_tokens.bin"
LABELS_FILE = "sft_labels.bin"       # 1 = answer token (loss), 0 = prompt/padding
TOKENIZER_FILE = "tokenizer.json"
TEMPLATE_FILE = "chat_template.json"
MANIFEST_FILE = "manifest.json"

QUESTION_TEMPLATES = (
    "请解释成语《{word}》的意思",
    "成语《{word}》是什么意思",
    "请说出《{word}》的含义",
)


class SFTError(ValueError):
    """SFT corpus/template missing, malformed or tampered with."""


def _one_line(text: str) -> str:
    """Keep an answer on one line so the turn separator stays unambiguous."""
    out = [ch if ch.isprintable() and ch != "\n" else " " for ch in text]
    return " ".join("".join(out).split()).strip()


def _question_chars() -> str:
    return "".join(t.replace("{word}", "") for t in QUESTION_TEMPLATES)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def build(
    source_json: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    *,
    max_entries: int | None = None,
    chat_template: ChatTemplate | None = None,
    exclude_words: set[str] | None = None,
    clean_fn=None,
) -> dict[str, Any]:
    """Write sft_tokens.bin + sft_labels.bin + tokenizer + template + manifest.

    ``exclude_words`` drops word-level holdout rows; ``clean_fn`` (shared with
    eval-3) normalises the answer so train and eval use byte-identical text.

    ``sft_labels.bin`` is a uint8 stream parallel to the uint16 token stream:
    1 for answer tokens (the only positions that contribute to loss), 0 for
    question / role-marker / BOS / EOS tokens. This gives a true answer-side
    loss instead of diluting the signal with the highly repetitive question.
    """
    from ..split import answer_for_row

    cleaner = clean_fn or answer_for_row
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(tokenizer_path)
    tmpl = chat_template or ChatTemplate.default()
    tmpl.validate_chars(tok)  # role markers must exist in the frozen vocab
    missing = sorted({ch for ch in _question_chars()
                      if ch not in tok.token_to_id})
    if missing:
        raise SFTError(f"question template characters missing from vocab: {missing!r}")

    src = Path(source_json)
    if not src.is_file():
        raise SFTError(f"source JSON not found: {src}")
    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SFTError("source JSON must be an array")

    ids: list[int] = []
    labels: list[int] = []
    used = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        word = (row.get("word") or "").strip()
        if exclude_words and word in exclude_words:
            continue
        answer = cleaner(row)
        if not word or not answer:
            continue
        question = QUESTION_TEMPLATES[used % len(QUESTION_TEMPLATES)].format(word=word)

        # Encode question-only and full turn to locate the answer span exactly
        # in token space: everything before the answer suffix gets label 0.
        q_ids = tok.encode(tmpl.training_text(question, ""))
        full_ids = tok.encode(tmpl.training_text(question, answer))
        if not full_ids[:len(q_ids)] == q_ids:
            raise SFTError("answer is not a clean suffix of the training turn")
        ans_ids = full_ids[len(q_ids):]
        if not ans_ids:
            continue

        ids.append(BOS_ID)
        labels.append(0)
        ids.extend(q_ids)
        labels.extend([0] * len(q_ids))
        ids.extend(ans_ids)
        labels.extend([1] * len(ans_ids))
        ids.append(EOS_ID)
        labels.append(0)
        used += 1
        if max_entries is not None and used >= max_entries:
            break

    if used < 1 or len(ids) < 1024:
        raise SFTError("not enough usable QA pairs to build an SFT corpus")

    arr = np.asarray(ids, dtype="<u2")
    lbl = np.asarray(labels, dtype=np.uint8)
    (out / TOKENS_FILE).write_bytes(arr.tobytes())
    (out / LABELS_FILE).write_bytes(lbl.tobytes())
    tok.save(out / TOKENIZER_FILE)
    tmpl.save(out / TEMPLATE_FILE)

    files: dict[str, Any] = {}
    for name in (TOKENS_FILE, LABELS_FILE, TOKENIZER_FILE, TEMPLATE_FILE):
        files[name] = {"sha256": _sha256_file(out / name),
                       "bytes": (out / name).stat().st_size}
    files[TOKENS_FILE]["tokens"] = int(arr.size)
    files[LABELS_FILE]["tokens"] = int(lbl.size)

    pinned = SOURCES.get("idiom", {})
    doc: dict[str, Any] = {
        "version": SFT_VERSION,
        "source": {
            "path": str(src),
            "sha256": _sha256_file(src),
            "license": pinned.get("license", "unknown"),
            "repo": pinned.get("repo", ""),
            "expected_sha256": pinned.get("sha256", ""),
        },
        "extraction": {
            "entries_used": used,
            "question_templates": list(QUESTION_TEMPLATES),
            "answer_source": "explanation | derivation | example",
            "format": "<bos> 用户问 Q \\n 助手答 A <eos>",
            "notes": "single-turn, deterministic, answers cleaned via autotrain.split.clean_answer",
            "answer_side_loss": True,
            "label_file": LABELS_FILE,
            "excluded_words": sorted(exclude_words) if exclude_words else [],
        },
        "tokenizer": {
            "type": tok.type,
            "vocab_size": tok.vocab_size,
            "sha256": tok.as_dict()["sha256"],
        },
        "chat_template": {
            "version": tmpl.name,
            "template_hash": tmpl.template_hash,
        },
        "files": files,
        "note": "SFT training corpus, NOT an eval set. evals/ and evalspec_hash "
                "are untouched; no wanxiang-eval-2 has been frozen.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    doc["corpus_id"] = hashlib.sha256(
        _canonical(sealed).encode("utf-8")
    ).hexdigest()[:16]
    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify(out_dir: str | Path) -> dict[str, Any]:
    """Re-check files, tokenizer, template and the corpus seal."""
    out = Path(out_dir)
    p = out / MANIFEST_FILE
    if not p.is_file():
        raise SFTError(f"manifest not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SFTError(f"invalid manifest JSON: {e}") from e
    if doc.get("version") != SFT_VERSION:
        raise SFTError(f"unsupported SFT corpus version: {doc.get('version')}")

    for name, rec in doc.get("files", {}).items():
        f = out / name
        if not f.is_file():
            raise SFTError(f"missing corpus file: {name}")
        if _sha256_file(f) != rec.get("sha256"):
            raise SFTError(f"checksum mismatch: {name}")
        if f.stat().st_size != rec.get("bytes"):
            raise SFTError(f"size mismatch: {name}")
        if name == TOKENS_FILE and f.stat().st_size // 2 != rec.get("tokens"):
            raise SFTError("token count mismatch")
        if name == LABELS_FILE:
            n_lbl = f.stat().st_size  # uint8
            n_tok = (out / TOKENS_FILE).stat().st_size // 2
            if n_lbl != rec.get("tokens") or n_lbl != n_tok:
                raise SFTError("label stream must be uint8 parallel to token stream")

    tok = load_tokenizer(out / TOKENIZER_FILE)
    if tok.as_dict()["sha256"] != doc["tokenizer"]["sha256"]:
        raise SFTError("tokenizer semantic checksum mismatch")
    tmpl = ChatTemplate.from_file(out / TEMPLATE_FILE)
    if tmpl.template_hash != doc["chat_template"]["template_hash"]:
        raise SFTError("chat template hash mismatch")
    tmpl.validate_chars(tok)

    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]
    if expected != doc.get("corpus_id"):
        raise SFTError("corpus_id seal mismatch (manifest was edited)")
    return doc
