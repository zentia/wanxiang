"""Versioned character-corpus builder and verifier (M2 minimal prototype).

Boundary with M1.5, kept deliberately strict:

* never writes into ``evals/`` (the frozen Markov corpora stay untouched);
* never changes ``VOCAB_SIZE`` (8192) or ``evalspec_hash``;
* the generated ``corpus_tokens.bin`` is a plain uint16 stream, so the existing
  ``TokenDataset`` / trainer / exporter can consume it with zero code changes.

A formal M2 would additionally freeze ``wanxiang-eval-2`` and re-seal the
evaluator. This prototype only versions the *training data*.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

CORPUS_VERSION = "wanxiang-m2-corpus-1"
TOKENS_FILE = "corpus_tokens.bin"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"
DEFAULT_FIELDS = ("word", "derivation", "example", "explanation")


class CorpusError(ValueError):
    """Corpus files or manifest are missing / tampered with."""


def _canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(
    source_json: str | Path,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    exclude_words: set[str] | None = None,
) -> tuple[str, int]:
    """Flatten selected Chinese text fields into one deterministic string.

    ``exclude_words`` holds out entire idiom words (see ``autotrain.split``)
    so a word-level holdout never leaks into the pretrain stream. The returned
    count is the number of rows actually written.
    """
    p = Path(source_json)
    with open(p, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise CorpusError("source JSON must be an array")
    parts: list[str] = []
    written = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if exclude_words and str(row.get("word", "")).strip() in exclude_words:
            continue
        for key in fields:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        parts.append("\n")
        written += 1
    return clean_text("".join(parts)), written


def clean_text(raw: str) -> str:
    """Normalise whitespace, drop control characters, keep visible Unicode."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in raw:
        if ch == "\n":
            out.append("\n")
        elif ch == "\t":
            out.append(" ")
        elif ch.isprintable():
            out.append(ch)
    text = "".join(out)
    # collapse runs of spaces and blank lines
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip() + "\n"


def build(
    source_json: str | Path,
    out_dir: str | Path,
    *,
    source_meta: dict[str, Any] | None = None,
    vocab_size: int = 8192,
    max_chars: int | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    exclude_words: set[str] | None = None,
    tokenizer_kind: str = "char",
) -> dict[str, Any]:
    """Build tokenizer + uint16 token stream + sealed manifest.

    With ``exclude_words`` the tokenizer is still built from the FULL text
    (so the frozen vocab is byte-identical to the original pipeline and the
    eval-3 holdout words remain encodable), but the written token stream
    excludes those words entirely -- no holdout text ever reaches training.

    ``tokenizer_kind`` selects ``char`` (one id per character) or ``bpe``
    (character-preserving BPE with deterministic frequent-pair merges). Both
    fill the same ``vocab_size`` budget so the model and bundle contracts do
    not change; the only difference is encoding granularity.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_text, full_entries = extract_text(source_json, fields)
    text, entries = (extract_text(source_json, fields, exclude_words)
                     if exclude_words else (full_text, full_entries))
    if len(text) < 1024:
        raise CorpusError("extracted text is too short to be a corpus")

    # Only stable provenance fields enter the seal: a "cached" vs "downloaded"
    # fetch status must never change corpus_id.
    stable_keys = ("name", "url", "repo", "license", "license_url",
                   "sha256", "bytes", "description")
    provenance = ({k: source_meta[k] for k in stable_keys if k in source_meta}
                  if source_meta else {"path": str(source_json)})

    tok_source = {"corpus_version": CORPUS_VERSION,
                  "source": provenance.get("name", "file")}
    if tokenizer_kind == "char":
        tokenizer = CharTokenizer.build_from_text(full_text, vocab_size=vocab_size,
                                                  source=tok_source)
    elif tokenizer_kind == "bpe":
        tokenizer = BPETokenizer.build_from_text(full_text, vocab_size=vocab_size,
                                                 source=tok_source)
    else:
        raise CorpusError(f"unsupported tokenizer_kind: {tokenizer_kind!r}")
    written = text if max_chars is None else text[:max_chars]
    ids = tokenizer.encode(written)
    if len(ids) < 1024:
        raise CorpusError("encoded token stream is too short")
    arr = np.asarray(ids, dtype="<u2")  # explicit little-endian uint16
    (out / TOKENS_FILE).write_bytes(arr.tobytes())
    tokenizer.save(out / TOKENIZER_FILE)

    files: dict[str, Any] = {}
    for name in (TOKENS_FILE, TOKENIZER_FILE):
        p = out / name
        files[name] = {"sha256": _sha256_bytes(p), "bytes": p.stat().st_size}
    files[TOKENS_FILE]["tokens"] = int(arr.size)

    doc: dict[str, Any] = {
        "version": CORPUS_VERSION,
        "source": provenance,
        "extraction": {
            "fields": list(fields),
            "entries": entries,
            "full_entries": full_entries,
            "text_chars": len(text),
            "written_chars": len(written),
            "cleaned": "whitespace normalised, control characters dropped",
            "excluded_words": sorted(exclude_words) if exclude_words else [],
            "holdout_note": "excluded word-level holdout; tokenizer built from full text",
        },
        "tokenizer": {
            "type": tokenizer.type,
            "vocab_size": vocab_size,
            "used_chars": tokenizer.used_chars,
            "n_merges": tokenizer.n_merges if tokenizer.type == "bpe" else 0,
            "sha256": tokenizer.as_dict()["sha256"],
            "eos_token_id": tokenizer.eos_token_id,
        },
        "split": {
            "val_fraction": 0.02,
            "note": "TokenDataset splits the tail 2% as validation; "
                    "the file itself is a single flat uint16 stream.",
        },
        "files": files,
        "note": "M2 prototype corpus. NOT an eval set: evals/ stays frozen and "
                "evalspec_hash is unchanged. A formal M2 must freeze wanxiang-eval-2.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    # corpus_id is a seal over everything except itself and the timestamp.
    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    doc["corpus_id"] = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]

    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify(out_dir: str | Path) -> dict[str, Any]:
    """Re-check every file and the seal. Raises ``CorpusError`` on any drift."""
    out = Path(out_dir)
    p = out / MANIFEST_FILE
    if not p.is_file():
        raise CorpusError(f"manifest not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorpusError(f"invalid manifest JSON: {e}") from e
    if doc.get("version") != CORPUS_VERSION:
        raise CorpusError(f"unsupported corpus version: {doc.get('version')}")

    for name, rec in doc.get("files", {}).items():
        f = out / name
        if not f.is_file():
            raise CorpusError(f"missing corpus file: {name}")
        if _sha256_bytes(f) != rec.get("sha256"):
            raise CorpusError(f"checksum mismatch: {name}")
        if f.stat().st_size != rec.get("bytes"):
            raise CorpusError(f"size mismatch: {name}")
        if name == TOKENS_FILE:
            n = f.stat().st_size // 2
            if n != rec.get("tokens"):
                raise CorpusError("token count mismatch")

    tok = load_tokenizer(out / TOKENIZER_FILE)  # also checks its own sha256
    if tok.as_dict()["sha256"] != doc["tokenizer"]["sha256"]:
        raise CorpusError("tokenizer semantic checksum mismatch")
    if tok.vocab_size != doc["tokenizer"]["vocab_size"]:
        raise CorpusError("tokenizer vocab_size mismatch")
    if tok.type != doc["tokenizer"]["type"]:
        raise CorpusError("tokenizer type mismatch")

    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]
    if expected != doc.get("corpus_id"):
        raise CorpusError("corpus_id seal mismatch (manifest was edited)")
    return doc
