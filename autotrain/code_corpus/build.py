"""Build a frozen code corpus: source jsonl -> train/holdout uint16 streams.

Output layout is byte-compatible with ``autotrain.corpus.pipeline.verify`` and
``autotrain.eval_lm`` so both existing tools work on the new corpus unchanged:

    <out>/
      train_tokens.bin
      holdout_tokens.bin
      tokenizer.json      BPETokenizer-compatible vocab (type="bpe")
      manifest.json       provenance-free, sealed with corpus_id

The general-purpose BPE learner only mines n-grams inside CJK runs, which
would leave ASCII code as a pure character model. ``learn_code_bpe`` learns
identifiers, in-word fragments, symbol runs and indentation instead, but emits
the SAME serialisation format, so ``load_tokenizer()``, the inference server
and the bundle contract all keep working.
"""

from __future__ import annotations

import array
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus.pipeline import (
    CORPUS_VERSION,
    HOLDOUT_FILE,
    MANIFEST_FILE,
    TOKENIZER_FILE,
    TRAIN_FILE,
    split_documents,
)
from ..inference.tokenizer import (
    SPECIAL_TOKENS,
    BPETokenizer,
    CharTokenizer,
    load_tokenizer,
)

MAX_PIECE_LEN = 12
DEFAULT_VOCAB_SIZE = 8192

# Manifests must stay free of absolute filesystem paths (anonymised corpora
# are derived from a private local source tree whose location is never stored).
_ABS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\[^\\]+\\")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,15}")
_SYMBOL_RE = re.compile(r"[^\w\s]{2,4}")
_INDENT_RE = re.compile(r"(?m)^([ \t]{2,16})")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer learning.
# ---------------------------------------------------------------------------


def learn_code_bpe(
    full_text: str,
    sample_text: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    source: dict[str, Any] | None = None,
) -> BPETokenizer:
    """Deterministic code-oriented subword vocab, BPETokenizer-compatible.

    Every character in the FULL corpus is a base token (so nothing is ever
    un-encodable). The remaining slots go to frequent identifiers, short
    in-word fragments, symbol runs and indentation strings learned from a
    bounded sample, ranked by (-frequency, -length, lexicographic).
    """
    ranked = sorted(Counter(full_text).items(), key=lambda kv: (-kv[1], kv[0]))
    base_chars = [ch for ch, _ in ranked]
    n_base = len(SPECIAL_TOKENS) + len(base_chars)
    if n_base > vocab_size:
        raise ValueError(
            f"corpus has {len(base_chars)} distinct chars, exceeds vocab_size={vocab_size}"
        )
    n_slots = vocab_size - n_base

    # Ranking everything by raw frequency lets 2-char in-word windows drown
    # out whole identifiers (they occur in almost every word). Select in three
    # stages instead: complete identifiers first, then symbols/indentation,
    # then whatever budget remains goes to short fragments.
    word_freq: Counter[str] = Counter()
    frag_freq: Counter[str] = Counter()
    symbol_freq: Counter[str] = Counter()
    words = [m.group() for m in _IDENT_RE.finditer(sample_text)]
    for w in words:
        if 2 <= len(w) <= MAX_PIECE_LEN:
            word_freq[w] += 1
        for width in range(2, 7):
            for i in range(len(w) - width + 1):
                frag_freq[w[i:i + width]] += 1
    for m in _SYMBOL_RE.finditer(sample_text):
        symbol_freq[m.group()] += 1
    for m in _INDENT_RE.finditer(sample_text):
        symbol_freq[m.group(1)] += 1

    pieces: list[str] = []
    seen = set(base_chars)

    def _rank(counter: Counter[str]) -> list[str]:
        return sorted(counter, key=lambda x: (-counter[x], -len(x), x))

    def _add(g: str) -> bool:
        """Return False once the vocab budget is exhausted."""
        if len(pieces) >= n_slots:
            return False
        if 2 <= len(g) <= MAX_PIECE_LEN and "\n" not in g and g not in seen:
            pieces.append(g)
            seen.add(g)
        return True

    # Stage 1: complete identifiers / keywords get ~45% of the budget.
    word_budget = int(n_slots * 0.45)
    for g in _rank(word_freq):
        if len(pieces) >= word_budget or not _add(g):
            break
    # Stage 2: operators and indentation runs (few, but high value).
    for g in _rank(symbol_freq):
        if not _add(g):
            break
    # Stage 3: short fragments fill the remainder for unseen-name coverage.
    for g in _rank(frag_freq):
        if not _add(g):
            break

    vocab = list(SPECIAL_TOKENS) + base_chars + pieces
    vocab += [f"<unused_{i}>" for i in range(vocab_size - len(vocab))]
    return BPETokenizer(vocab, pieces, vocab_size=vocab_size, source=source or {})


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("source jsonl record has no text")
            docs.append(rec)
    if len(docs) < 10:
        raise ValueError("fewer than 10 documents in source jsonl")
    return docs


def _encode_documents(
    docs: list[dict[str, Any]], indices: list[int], tokenizer: Any
) -> list[int]:
    """Encode document-by-document, with EOS as an explicit file boundary."""
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for i in indices:
        ids.extend(tokenizer.encode(docs[i]["text"]))
        if eos is not None:
            ids.append(int(eos))
    return ids


def _encode_documents_to_file(
    docs: list[dict[str, Any]], indices: list[int], tokenizer: Any,
    out_path: Path,
) -> int:
    """Stream uint16 tokens to disk in bounded memory.

    A 512 MiB corpus yields ~200M tokens; holding them in a Python list of ints
    would cost several GiB. ``array('H')`` keeps the buffer compact and the
    emitted bytes are identical to the old ``np.asarray(..., dtype='<u2')``.
    """
    eos = tokenizer.eos_token_id
    total = 0
    with out_path.open("wb") as f:
        buf = array.array("H")
        for i in indices:
            buf.extend(tokenizer.encode(docs[i]["text"]))
            if eos is not None:
                buf.append(int(eos))
            if len(buf) >= 1_000_000:
                if sys.byteorder != "little":
                    buf.byteswap()
                f.write(buf.tobytes())
                total += len(buf)
                buf = array.array("H")
        if buf:
            if sys.byteorder != "little":
                buf.byteswap()
            f.write(buf.tobytes())
            total += len(buf)
    return total


def build(
    source_jsonl: str | Path,
    out_dir: str | Path,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    tokenizer_kind: str = "bpe",
    holdout_fraction: float = 0.05,
    seed: int = 42,
    tokenizer_sample_chars: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Write frozen code corpus + sealed manifest. Returns the manifest dict."""
    src = Path(source_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = _read_jsonl(src)
    train_idx, holdout_idx = split_documents(
        len(docs), holdout_fraction=holdout_fraction, seed=seed
    )

    full_text = "".join(d["text"] for d in docs)
    train_text = "".join(docs[i]["text"] for i in train_idx)
    holdout_text = "".join(docs[i]["text"] for i in holdout_idx)
    if len(train_text) < 1024:
        raise ValueError("train text too short to be a corpus")
    if len(holdout_text) < 1024:
        raise ValueError("holdout text too short for a meaningful eval set")

    tok_source = {
        "corpus_version": CORPUS_VERSION,
        "tokenizer_kind": tokenizer_kind,
        "seed": seed,
        "note": "code-oriented subword vocab; no source paths recorded",
    }
    if tokenizer_kind == "bpe":
        tokenizer = learn_code_bpe(
            full_text, full_text[:tokenizer_sample_chars],
            vocab_size=vocab_size, source=tok_source,
        )
    elif tokenizer_kind == "char":
        tokenizer = CharTokenizer.build_from_text(
            full_text, vocab_size=vocab_size, source=tok_source
        )
    else:
        raise ValueError(f"unsupported tokenizer_kind: {tokenizer_kind!r}")

    print(f"[build] encoding {len(train_idx)} train / {len(holdout_idx)} holdout docs")
    n_train = _encode_documents_to_file(docs, train_idx, tokenizer, out / TRAIN_FILE)
    n_holdout = _encode_documents_to_file(
        docs, holdout_idx, tokenizer, out / HOLDOUT_FILE
    )
    if n_train < 1024 or n_holdout < 64:
        raise ValueError("token stream too short")
    tokenizer.save(out / TOKENIZER_FILE)

    files: dict[str, Any] = {}
    for name in (TRAIN_FILE, HOLDOUT_FILE, TOKENIZER_FILE):
        p = out / name
        files[name] = {"sha256": _sha256_bytes(p), "bytes": p.stat().st_size}
    files[TRAIN_FILE]["tokens"] = n_train
    files[HOLDOUT_FILE]["tokens"] = n_holdout

    # Anonymous sampling provenance (no root path, no filenames).
    sampling: dict[str, Any] = {}
    summary_path = src.parent / "summary.json"
    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        sampling = {
            "sampling_version": s.get("sampling_version"),
            "seed": s.get("seed"),
            "max_bytes": s.get("max_bytes"),
            "sample_files": s.get("sample_files"),
            "documents": s.get("documents"),
            "by_lang": s.get("by_lang"),
            "source_sha256": s.get("source_sha256"),
        }

    by_lang: Counter[str] = Counter(d.get("lang", "?") for d in docs)
    doc: dict[str, Any] = {
        "version": CORPUS_VERSION,
        "source": {
            "name": "local-code",
            "license": "proprietary-local",
            "description": "anonymised local source snapshot; paths omitted",
            "pin": "code-corpus-anon-1",
        },
        "extraction": {
            "documents": len(docs),
            "train_documents": len(train_idx),
            "holdout_documents": len(holdout_idx),
            "train_chars": len(train_text),
            "holdout_chars": len(holdout_text),
            "by_lang": dict(sorted(by_lang.items())),
            "cleaned": "line endings normalised; newlines / tabs / indentation "
                       "PRESERVED; control chars and blank-line runs collapsed",
            "document_boundary": "EOS token between files",
        },
        "split": {
            "method": "seeded-document-index-permutation",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "holdout_indices": holdout_idx,
            "note": "holdout split by FILE before concatenation; a wrapping "
                    "training cursor can never reach evaluation code",
        },
        "tokenizer": {
            "type": tokenizer.type,
            "vocab_size": vocab_size,
            "used_chars": tokenizer.used_chars,
            "n_merges": tokenizer.n_merges if tokenizer.type == "bpe" else 0,
            "sha256": tokenizer.as_dict()["sha256"],
            "eos_token_id": tokenizer.eos_token_id,
            "sample_chars": tokenizer_sample_chars,
            "note": "code-oriented subwords (identifiers / fragments / symbols "
                    "/ indentation); all corpus chars kept as base tokens",
        },
        "sampling": sampling,
        "files": files,
        "note": "Frozen CODE corpus. NOT comparable with the wiki LM corpus "
                "(different corpus_id / tokenizer / holdout -> different eval_lm_hash).",
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
    """Delegate to the general-purpose verifier so both corpora share one seal."""
    from ..corpus.pipeline import verify as _verify

    result = _verify(out_dir)
    # Extra invariant: no absolute filesystem path may enter a manifest.
    blob = json.dumps(result, ensure_ascii=False)
    if _ABS_PATH_RE.search(blob):
        raise ValueError("manifest leaked an absolute filesystem path")
    load_tokenizer(Path(out_dir) / TOKENIZER_FILE)
    return result
