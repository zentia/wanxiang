"""Tokenizer boundary for the inference service.

M1 bundles have no tokenizer: a token id is just a uint16 integer from a
flat stream, with no mapping to or from text. The service makes that fact
explicit instead of pretending the model can read or write language.

M2 replaces ``IdentityTokenizer`` with a real tokenizer implementing the
same ``Tokenizer`` protocol. The engine and the HTTP endpoints do not change:
``/v1/completions`` simply stops rejecting text prompts and starts returning
decoded text.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TokenizerUnavailable(RuntimeError):
    """Raised when text input/output is requested but no tokenizer exists."""


@runtime_checkable
class Tokenizer(Protocol):
    supports_text: bool
    vocab_size: int
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class IdentityTokenizer:
    """M1: token ids in, token ids out. ``supports_text`` is deliberately False."""

    type = "identity"

    def __init__(self, vocab_size: int, eos_token_id: int | None = None):
        self.supports_text = False
        self.vocab_size = int(vocab_size)
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        raise TokenizerUnavailable(
            "M1 bundles have no tokenizer (uint16 flat token stream); text prompts "
            "are unsupported until M2. Use /v1/token-completions with prompt_ids."
        )

    def decode(self, ids: list[int]) -> str:
        # There is no id -> text mapping. Return an empty string and let the
        # server report ``text: null`` plus the raw token ids.
        return ""


def validate_token_ids(ids: list[int], vocab_size: int) -> None:
    """Reject non-integer / out-of-range ids before they reach the model."""
    if not isinstance(ids, list) or not ids:
        raise ValueError("prompt_ids must be a non-empty list of integers")
    for t in ids:
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError("prompt_ids must contain integers only")
        if not 0 <= t < vocab_size:
            raise ValueError(f"token id {t} outside vocab range [0, {vocab_size})")


# ---------------------------------------------------------------------------
# M2 minimal prototype: character-level tokenizer.
#
# The M1.5 frozen spec uses VOCAB_SIZE = 8192, so the M2 prototype keeps that
# exact budget. Chinese has a few thousand common characters, which fits
# comfortably at character granularity (no BPE needed yet).
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class CharTokenizerError(ValueError):
    """Tokenizer file is missing, malformed or has been tampered with."""


def _canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class CharTokenizer:
    """Deterministic character-level tokenizer implementing ``Tokenizer``.

    ``supports_text=True``: the inference service can now encode text prompts
    and decode generated ids back into characters. It does NOT make the model
    a language model -- that needs real pretraining + SFT (see M2.md).
    """

    type = "char"

    def __init__(self, id_to_token: list[str], *, vocab_size: int = 8192,
                 source: dict[str, Any] | None = None):
        if len(id_to_token) != vocab_size:
            raise CharTokenizerError(
                f"vocab has {len(id_to_token)} entries, expected {vocab_size}"
            )
        if tuple(id_to_token[:4]) != SPECIAL_TOKENS:
            raise CharTokenizerError("first four ids must be the special tokens")
        if len(set(id_to_token)) != len(id_to_token):
            raise CharTokenizerError("vocab contains duplicate tokens")
        self.id_to_token = list(id_to_token)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.vocab_size = int(vocab_size)
        self.supports_text = True
        self.eos_token_id: int | None = EOS_ID
        self.bos_token_id: int | None = BOS_ID
        self.pad_token_id = PAD_ID
        self.unk_token_id = UNK_ID
        self.source = source or {}

    # ---------------------------------------------------------------- build
    @classmethod
    def build_from_text(cls, text: str, *, vocab_size: int = 8192,
                        source: dict[str, Any] | None = None) -> "CharTokenizer":
        """Frequency-ranked characters, deterministically ordered.

        Ties break by Unicode code point so the same corpus always produces
        the same vocab. Unused ids are padded with ``<unused_N>`` and decode
        to nothing (they are still valid model outputs).
        """
        if vocab_size <= len(SPECIAL_TOKENS):
            raise CharTokenizerError("vocab_size too small for special tokens")
        counts = Counter(text)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        chars = [ch for ch, _ in ranked[:vocab_size - len(SPECIAL_TOKENS)]]
        vocab = list(SPECIAL_TOKENS) + chars
        vocab += [f"<unused_{i}>" for i in range(vocab_size - len(vocab))]
        return cls(vocab, vocab_size=vocab_size, source=source)

    @property
    def used_chars(self) -> int:
        return sum(1 for t in self.id_to_token[4:] if not t.startswith("<unused_"))

    # ---------------------------------------------------------------- serde
    def as_dict(self, *, with_hash: bool = True) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "type": self.type,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID,
            },
            "id_to_token": self.id_to_token,
            "source": self.source,
            "note": "Character-level M2 prototype; unused ids decode to nothing.",
        }
        if with_hash:
            doc["sha256"] = hashlib.sha256(_canonical(doc).encode("utf-8")).hexdigest()
        return doc

    def save(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = self.as_dict()
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return doc

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        p = Path(path)
        if not p.is_file():
            raise CharTokenizerError(f"tokenizer file not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CharTokenizerError(f"invalid tokenizer JSON: {e}") from e
        if doc.get("type") != cls.type:
            raise CharTokenizerError("tokenizer type must be 'char'")
        declared = doc.get("sha256")
        if not declared:
            raise CharTokenizerError("tokenizer file has no sha256")
        body = {k: v for k, v in doc.items() if k != "sha256"}
        actual = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        if actual != declared:
            raise CharTokenizerError("tokenizer checksum mismatch (file was modified)")
        return cls(doc["id_to_token"], vocab_size=doc["vocab_size"],
                   source=doc.get("source", {}))

    # ---------------------------------------------------------------- text
    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        ids.extend(self.token_to_id.get(ch, UNK_ID) for ch in text)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if not 0 <= i < self.vocab_size:
                continue
            tok = self.id_to_token[i]
            if tok.startswith("<unused_"):
                continue
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)


# ---------------------------------------------------------------------------
# M3 prototype: byte-free character-level BPE.
#
# Design goal: change ONLY the encoding granularity, keeping vocab_size=8192
# and the model/bundle/eval contracts byte-identical to the char prototype so
# an A/B on the frozen eval-3 set isolates the effect of subword tokenization.
#
# Every single character seen in the corpus is kept as a base token (so the
# 150 word-level holdout idioms and the chat template markers remain exactly
# encodable). The remaining budget is spent on deterministically-learned BPE
# merges (frequent adjacent pairs -> one token), which shortens common words
# like 形容/比喻/意思 into single ids and improves compositional generalisation.
# ---------------------------------------------------------------------------

import re  # noqa: E402

_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]+|[^\u4e00-\u9fff\s]+|\s+")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
MAX_MERGE_CHARS = 4  # keep subword tokens word/phrase-sized, never whole sentences


def split_segments(text: str) -> list[str]:
    """Split text into CJK runs, non-CJK runs, and whitespace.

    BPE merges are learned and applied WITHIN a segment, never across its
    boundary. This keeps encode() fast (short segments) and deterministic.
    """
    return _SEGMENT_RE.findall(text)


class BPETokenizerError(ValueError):
    """BPE tokenizer file is missing, malformed or has been tampered with."""


class BPETokenizer:
    """Deterministic subword tokenizer implementing ``Tokenizer``.

    A character-preserving subword model: every corpus character is a base
    token (so holdout idioms and chat markers stay exactly encodable), plus
    the most frequent 2-4 character n-grams as subword tokens. Encoding uses
    greedy longest-match. This is O(N) to learn and O(L) to encode -- no
    iterative re-tokenisation -- and keeps subwords word/phrase sized.

    vocab layout: ``<pad> <unk> <bos> <eos>`` (ids 0-3, same as the char
    prototype), then all corpus chars by descending frequency, then subword
    pieces by descending frequency, then unused padding. ``vocab_size`` is
    unchanged at 8192 so the model, bundle and eval contracts do not change.
    """

    type = "bpe"

    def __init__(
        self,
        id_to_token: list[str],
        pieces: list[str],
        *,
        vocab_size: int = 8192,
        source: dict[str, Any] | None = None,
    ):
        if len(id_to_token) != vocab_size:
            raise BPETokenizerError(
                f"vocab has {len(id_to_token)} entries, expected {vocab_size}"
            )
        if tuple(id_to_token[:4]) != SPECIAL_TOKENS:
            raise BPETokenizerError("first four ids must be the special tokens")
        if len(set(id_to_token)) != len(id_to_token):
            raise BPETokenizerError("vocab contains duplicate tokens")
        self.id_to_token = list(id_to_token)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pieces = list(pieces)
        for piece in self.pieces:
            if piece not in self.token_to_id:
                raise BPETokenizerError(f"subword piece {piece!r} has no vocab token")
        # Longest-match lookup: pieces grouped by length, longest first.
        self._pieces_by_len: dict[int, set[str]] = {}
        for p in self.pieces:
            self._pieces_by_len.setdefault(len(p), set()).add(p)
        self._max_piece_len = max(self._pieces_by_len, default=1)
        self.vocab_size = int(vocab_size)
        self.supports_text = True
        self.eos_token_id: int | None = EOS_ID
        self.bos_token_id: int | None = BOS_ID
        self.pad_token_id = PAD_ID
        self.unk_token_id = UNK_ID
        self.source = source or {}

    @property
    def used_chars(self) -> int:
        return sum(
            1 for t in self.id_to_token[4:]
            if not t.startswith("<unused_") and len(t) == 1
        )

    @property
    def n_merges(self) -> int:
        """Number of subword pieces (kept as `n_merges` for manifest compat)."""
        return len(self.pieces)

    # ---------------------------------------------------------------- build
    @classmethod
    def build_from_text(
        cls,
        text: str,
        *,
        vocab_size: int = 8192,
        source: dict[str, Any] | None = None,
    ) -> "BPETokenizer":
        """Learn char + frequent-ngram vocab from ``text`` in O(N).

        Deterministic: chars ranked by (-frequency, codepoint); n-gram pieces
        by (-frequency, -length, piece) so an identical corpus always yields a
        byte-identical vocab and sha256.
        """
        if vocab_size <= len(SPECIAL_TOKENS):
            raise BPETokenizerError("vocab_size too small for special tokens")

        # 1) base characters (all of them)
        counts = Counter(text)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        chars = [ch for ch, _ in ranked]
        n_base = len(SPECIAL_TOKENS) + len(chars)
        if n_base > vocab_size:
            raise BPETokenizerError(
                f"corpus has {len(chars)} distinct chars, exceeds vocab_size={vocab_size}"
            )

        # 2) frequent n-grams (2..MAX_MERGE_CHARS), counted in one pass over
        # each CJK segment only (subwords never cross segment boundaries).
        ngram_freq: Counter[str] = Counter()
        for seg in split_segments(text):
            if not _CJK_RE.match(seg):
                continue
            n = len(seg)
            for width in range(2, min(MAX_MERGE_CHARS, n) + 1):
                for i in range(n - width + 1):
                    ngram_freq[seg[i:i + width]] += 1

        n_slots = vocab_size - n_base
        # Dedup overlapping pieces greedily: prefer longer, then more frequent.
        candidates = sorted(
            ngram_freq,
            key=lambda g: (-ngram_freq[g], -len(g), g),
        )
        pieces: list[str] = []
        seen_chars = set(chars)
        for g in candidates:
            if len(pieces) >= n_slots:
                break
            if g in seen_chars:
                continue
            pieces.append(g)
            seen_chars.add(g)

        vocab = list(SPECIAL_TOKENS) + chars + pieces
        vocab += [f"<unused_{i}>" for i in range(vocab_size - len(vocab))]
        return cls(vocab, pieces, vocab_size=vocab_size, source=source)

    # ---------------------------------------------------------------- serde
    def as_dict(self, *, with_hash: bool = True) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "type": self.type,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID,
            },
            "id_to_token": self.id_to_token,
            "pieces": self.pieces,
            "source": self.source,
            "note": "Character-preserving subword tokenizer: all corpus chars "
                    "kept, remaining slots are the most frequent 2-4 char "
                    "n-grams; greedy longest-match encoding.",
        }
        if with_hash:
            doc["sha256"] = hashlib.sha256(_canonical(doc).encode("utf-8")).hexdigest()
        return doc

    def save(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = self.as_dict()
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return doc

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        p = Path(path)
        if not p.is_file():
            raise BPETokenizerError(f"tokenizer file not found: {p}")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BPETokenizerError(f"invalid tokenizer JSON: {e}") from e
        if doc.get("type") != cls.type:
            raise BPETokenizerError("tokenizer type must be 'bpe'")
        declared = doc.get("sha256")
        if not declared:
            raise BPETokenizerError("tokenizer file has no sha256")
        body = {k: v for k, v in doc.items() if k != "sha256"}
        actual = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        if actual != declared:
            raise BPETokenizerError("tokenizer checksum mismatch (file was modified)")
        return cls(
            doc["id_to_token"],
            doc.get("pieces", []),
            vocab_size=doc["vocab_size"],
            source=doc.get("source", {}),
        )

    # ---------------------------------------------------------------- text
    def _encode_segment(self, seg: str) -> list[str]:
        # Greedy longest-match within a segment.
        out: list[str] = []
        i = 0
        n = len(seg)
        while i < n:
            matched = None
            upper = min(self._max_piece_len, n - i)
            for width in range(upper, 1, -1):
                if seg[i:i + width] in self._pieces_by_len.get(width, ()):
                    matched = seg[i:i + width]
                    break
            if matched is not None:
                out.append(matched)
                i += len(matched)
            else:
                out.append(seg[i])
                i += 1
        return out

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for seg in split_segments(text):
            for tok in self._encode_segment(seg):
                ids.append(self.token_to_id.get(tok, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if not 0 <= i < self.vocab_size:
                continue
            tok = self.id_to_token[i]
            if tok.startswith("<unused_"):
                continue
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer by its declared ``type`` (char / bpe).

    Single dispatch point so the inference server, evaluator and corpus
    verifier never hard-code one tokenizer class. Unknown or missing types
    fail loudly instead of silently falling back.
    """
    p = Path(path)
    if not p.is_file():
        raise CharTokenizerError(f"tokenizer file not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CharTokenizerError(f"invalid tokenizer JSON: {e}") from e
    kind = doc.get("type")
    if kind == "char":
        return CharTokenizer.from_file(p)
    if kind == "bpe":
        return BPETokenizer.from_file(p)
    raise CharTokenizerError(f"unsupported tokenizer type: {kind!r}")
