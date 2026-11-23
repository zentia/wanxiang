"""Versioned general-purpose corpus builder and verifier.

Turns a sealed source jsonl (one document per line, with a ``text`` field)
into the flat uint16 token streams the unchanged trainer consumes:

    <out>/
      train_tokens.bin    uint16 little-endian, for training
      holdout_tokens.bin  uint16 little-endian, frozen LM eval set (never trained)
      tokenizer.json      char / bpe tokenizer (versioned, checksummed)
      manifest.json       provenance + deterministic split + file seals

Determinism rules:
* holdout is split by document index (seeded permutation), not by token
  offset -- so a longer training run can never wrap into the holdout;
* tokenizer is built from the FULL text (train + holdout) so every holdout
  character is encodable, but the holdout token stream is never fed to the
  trainer;
* corpus_id seals everything except itself and the timestamp.

No idiom / game / retrieval assumptions. Natural language in, tokens out.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

CORPUS_VERSION = "wanxiang-corpus-lm-1"
TRAIN_FILE = "train_tokens.bin"
HOLDOUT_FILE = "holdout_tokens.bin"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"


class CorpusError(ValueError):
    """Corpus files or manifest are missing / tampered with."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(raw: str) -> str:
    """Normalise whitespace and drop control characters; keep visible Unicode."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in raw:
        if ch in ("\n", "\t"):
            out.append(" ")
        elif ch.isprintable():
            out.append(ch)
    text = "".join(out)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip() + "\n"


def read_documents(source_jsonl: str | Path) -> list[str]:
    """Read cleaned document texts from a source jsonl (one object per line)."""
    p = Path(source_jsonl)
    if not p.is_file():
        raise CorpusError(f"source jsonl not found: {p}")
    docs: list[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise CorpusError(f"invalid source line: {e}") from e
            text = rec.get("text") if isinstance(rec, dict) else None
            if isinstance(text, str):
                cleaned = clean_text(text)
                if cleaned.strip():
                    docs.append(cleaned)
    if not docs:
        raise CorpusError("no usable documents in source jsonl")
    return docs


def split_documents(
    n_docs: int, *, holdout_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic train/holdout document index split (seeded permutation).

    Splitting by document (before concatenation) guarantees the holdout text
    can never be reached by a training cursor that wraps around.
    """
    if n_docs < 2:
        raise CorpusError("need at least 2 documents to split train/holdout")
    if not 0.0 < holdout_fraction < 0.5:
        raise CorpusError("holdout_fraction must be in (0, 0.5)")
    idx = list(range(n_docs))
    random.Random(seed).shuffle(idx)
    n_holdout = max(1, int(n_docs * holdout_fraction))
    holdout = sorted(idx[:n_holdout])
    train = sorted(idx[n_holdout:])
    return train, holdout


def build(
    source_jsonl: str | Path,
    out_dir: str | Path,
    *,
    source_meta: dict[str, Any] | None = None,
    vocab_size: int = 8192,
    tokenizer_kind: str = "char",
    holdout_fraction: float = 0.05,
    seed: int = 42,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Build tokenizer + train/holdout token streams + sealed manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = read_documents(source_jsonl)
    train_idx, holdout_idx = split_documents(
        len(docs), holdout_fraction=holdout_fraction, seed=seed
    )
    full_text = "".join(docs)
    train_text = "".join(docs[i] for i in train_idx)
    holdout_text = "".join(docs[i] for i in holdout_idx)
    if max_chars is not None:
        train_text = train_text[:max_chars]

    if len(train_text) < 1024:
        raise CorpusError("train text too short to be a corpus")
    if len(holdout_text) < 128:
        raise CorpusError("holdout text too short for a meaningful eval set")

    tok_source = {
        "corpus_version": CORPUS_VERSION,
        "tokenizer_kind": tokenizer_kind,
        "seed": seed,
    }
    if tokenizer_kind == "char":
        tokenizer = CharTokenizer.build_from_text(full_text, vocab_size=vocab_size,
                                                  source=tok_source)
    elif tokenizer_kind == "bpe":
        tokenizer = BPETokenizer.build_from_text(full_text, vocab_size=vocab_size,
                                                 source=tok_source)
    else:
        raise CorpusError(f"unsupported tokenizer_kind: {tokenizer_kind!r}")

    train_ids = tokenizer.encode(train_text)
    holdout_ids = tokenizer.encode(holdout_text)
    if len(train_ids) < 1024:
        raise CorpusError("train token stream too short")
    if len(holdout_ids) < 64:
        raise CorpusError("holdout token stream too short")

    train_arr = np.asarray(train_ids, dtype="<u2")
    holdout_arr = np.asarray(holdout_ids, dtype="<u2")
    (out / TRAIN_FILE).write_bytes(train_arr.tobytes())
    (out / HOLDOUT_FILE).write_bytes(holdout_arr.tobytes())
    tokenizer.save(out / TOKENIZER_FILE)

    files: dict[str, Any] = {}
    for name in (TRAIN_FILE, HOLDOUT_FILE, TOKENIZER_FILE):
        p = out / name
        files[name] = {"sha256": _sha256_bytes(p), "bytes": p.stat().st_size}
    files[TRAIN_FILE]["tokens"] = int(train_arr.size)
    files[HOLDOUT_FILE]["tokens"] = int(holdout_arr.size)

    # Only stable provenance fields enter the seal.
    stable_keys = ("name", "dataset", "config", "split", "offset", "n_rows",
                   "sha256", "bytes", "rows", "license", "license_url",
                   "description", "pin")
    provenance = (
        {k: source_meta[k] for k in stable_keys if k in source_meta}
        if source_meta else {}
    )

    doc: dict[str, Any] = {
        "version": CORPUS_VERSION,
        "source": provenance,
        "extraction": {
            "documents": len(docs),
            "train_documents": len(train_idx),
            "holdout_documents": len(holdout_idx),
            "train_chars": len(train_text),
            "holdout_chars": len(holdout_text),
            "cleaned": "whitespace normalised, control characters dropped",
        },
        "split": {
            "method": "seeded-document-index-permutation",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "holdout_indices": holdout_idx,
            "note": "holdout split by document before concatenation; "
                    "training cursor wrap-around can never reach it",
        },
        "tokenizer": {
            "type": tokenizer.type,
            "vocab_size": vocab_size,
            "used_chars": tokenizer.used_chars,
            "n_merges": tokenizer.n_merges if tokenizer.type == "bpe" else 0,
            "sha256": tokenizer.as_dict()["sha256"],
            "eos_token_id": tokenizer.eos_token_id,
            "note": "built from FULL text (train+holdout) so holdout is encodable; "
                    "holdout tokens are never written to the train stream",
        },
        "files": files,
        "note": "General-purpose LM corpus. NOT comparable with eval2/3/4 "
                "(idiom answer-quality) or eval4 (retrieval) — this is raw "
                "next-token language modelling data.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    doc["corpus_id"] = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]

    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify(out_dir: str | Path) -> dict[str, Any]:
    """Re-check every file and the corpus seal. Raises ``CorpusError`` on drift."""
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
        if "tokens" in rec and f.stat().st_size // 2 != rec["tokens"]:
            raise CorpusError(f"token count mismatch: {name}")

    tok = load_tokenizer(out / TOKENIZER_FILE)  # also checks its own sha256
    if tok.as_dict()["sha256"] != doc["tokenizer"]["sha256"]:
        raise CorpusError("tokenizer semantic checksum mismatch")
    if tok.vocab_size != doc["tokenizer"]["vocab_size"]:
        raise CorpusError("tokenizer vocab_size mismatch")

    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]
    if expected != doc.get("corpus_id"):
        raise CorpusError("corpus_id seal mismatch (manifest was edited)")
    return doc
