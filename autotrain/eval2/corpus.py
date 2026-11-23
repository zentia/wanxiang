"""Build and verify the frozen wanxiang-eval-2 QA corpus.

Holdout construction (the part that prevents memorisation leaks):

* the SFT token stream is a flat sequence of ``<bos> … <eos>`` samples;
* SFT training consumed tokens [122880, 512000) of that stream;
* this frozen set takes samples whose **source offset > 800000**, i.e. far
  beyond anything the model ever saw during training;
* the negative control uses the same questions with deterministically
  shuffled references.

Everything is byte-checked; adding/modifying/removing a file fails verify.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.chat import ChatTemplate
from ..inference.tokenizer import BOS_ID, EOS_ID, CharTokenizer
from .spec import (CONTROL_ROTATION, EVAL2_VERSION, FROZEN,
                   MIN_SOURCE_OFFSET, QA_COUNT, TRAIN_CONSUMED_MAX_TOKEN,
                   _canonical)

EVAL2_DIR = Path(__file__).resolve().parents[2] / "evals2"
QA_FILE = "qa.jsonl"
CONTROL_FILE = "negative_control.jsonl"
MANIFEST_FILE = "manifest.json"
EXPECTED_FILES = (QA_FILE, CONTROL_FILE, MANIFEST_FILE)
WORD_RE = re.compile(r"《(.+?)》")


class Eval2CorpusError(ValueError):
    """Frozen eval-2 corpus missing, malformed or tampered with."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )


def freeze(
    sft_tokens: str | Path,
    tokenizer: CharTokenizer,
    chat_template: ChatTemplate,
    out_dir: str | Path | None = None,
    *,
    count: int = QA_COUNT,
    min_offset: int = MIN_SOURCE_OFFSET,
) -> dict[str, Any]:
    """Create the frozen QA set + negative control + sealed manifest."""
    out = Path(out_dir) if out_dir is not None else EVAL2_DIR
    out.mkdir(parents=True, exist_ok=True)
    chat_template.validate_chars(tokenizer)

    arr = np.fromfile(sft_tokens, dtype="<u2")
    starts = np.flatnonzero(arr == BOS_ID)
    ends = np.flatnonzero(arr == EOS_ID)
    if len(starts) != len(ends):
        raise Eval2CorpusError("BOS/EOS markers are unbalanced in the SFT stream")

    samples: list[dict[str, Any]] = []
    sep = "\n" + chat_template.role_prefix["assistant"]
    for s, e in zip(starts, ends):
        off = int(s)
        if off <= min_offset or e <= s:
            continue
        text = tokenizer.decode(arr[s + 1:e].tolist())
        if sep not in text:
            continue
        question, answer = text.split(sep, 1)
        question = question.removeprefix(chat_template.role_prefix["user"]).strip()
        answer = answer.strip()
        m = WORD_RE.search(question)
        if not m or not answer:
            continue
        samples.append({
            "word": m.group(1),
            "question": question,
            "reference": answer,
            "source_offset": off,
        })
        if len(samples) >= count:
            break
    if len(samples) < count:
        raise Eval2CorpusError(
            f"only {len(samples)} holdout samples available, need {count}"
        )

    control = [
        {
            "question": samples[i]["question"],
            "reference": samples[(i + CONTROL_ROTATION) % count]["reference"],
            "source_offset": samples[i]["source_offset"],
            "note": "shuffled reference: same question, wrong answer",
        }
        for i in range(count)
    ]

    _write_jsonl(out / QA_FILE, samples)
    _write_jsonl(out / CONTROL_FILE, control)

    doc: dict[str, Any] = {
        "format_version": EVAL2_VERSION,
        "frozen": FROZEN,
        "qa_count": count,
        "min_source_offset": min_offset,
        "train_consumed_max_token": TRAIN_CONSUMED_MAX_TOKEN,
        "control_rotation": CONTROL_ROTATION,
        "source": {
            "sft_tokens": str(sft_tokens),
            "sha256": _sha256_file(Path(sft_tokens)),
            "note": "QA pairs parsed back from SFT token stream, offsets > min_source_offset",
        },
        "tokenizer": {"type": tokenizer.type, "vocab_size": tokenizer.vocab_size,
                      "sha256": tokenizer.as_dict()["sha256"]},
        "chat_template": {"version": chat_template.name,
                          "template_hash": chat_template.template_hash},
        "files": {
            QA_FILE: {"sha256": _sha256_file(out / QA_FILE), "bytes": (out / QA_FILE).stat().st_size, "rows": count},
            CONTROL_FILE: {"sha256": _sha256_file(out / CONTROL_FILE), "bytes": (out / CONTROL_FILE).stat().st_size, "rows": count},
        },
        "note": "Frozen wanxiang-eval-2 measurement instrument. Never regenerate in place. "
                "M1.5 evals/ and evalspec_hash are deliberately untouched.",
    }
    doc["corpus_manifest_hash"] = hashlib.sha256(
        _canonical({k: v for k, v in doc.items() if k != "corpus_manifest_hash"}).encode("utf-8")
    ).hexdigest()
    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify_corpus(eval_dir: str | Path | None = None) -> dict[str, Any]:
    """Byte-check the frozen set. Any extra/missing/edited file fails."""
    d = Path(eval_dir) if eval_dir is not None else EVAL2_DIR
    if not d.is_dir():
        raise Eval2CorpusError(f"eval-2 corpus directory not found: {d}")
    actual = sorted(p.name for p in d.iterdir() if p.is_file())
    if actual != sorted(EXPECTED_FILES):
        raise Eval2CorpusError(
            f"eval-2 file set mismatch: expected {sorted(EXPECTED_FILES)}, got {actual}"
        )
    p = d / MANIFEST_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise Eval2CorpusError(f"invalid manifest JSON: {e}") from e
    if doc.get("format_version") != EVAL2_VERSION or doc.get("frozen") is not True:
        raise Eval2CorpusError("eval-2 corpus is not a frozen wanxiang-eval-2 set")

    for name, rec in doc["files"].items():
        f = d / name
        if _sha256_file(f) != rec["sha256"]:
            raise Eval2CorpusError(f"checksum mismatch: {name}")
        if f.stat().st_size != rec["bytes"]:
            raise Eval2CorpusError(f"size mismatch: {name}")
        if len(read_jsonl(f)) != rec["rows"]:
            raise Eval2CorpusError(f"row count mismatch: {name}")

    sealed = {k: v for k, v in doc.items() if k != "corpus_manifest_hash"}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()
    if expected != doc.get("corpus_manifest_hash"):
        raise Eval2CorpusError("corpus_manifest_hash seal mismatch (corpus was edited)")
    return doc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_qa(eval_dir: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d = Path(eval_dir) if eval_dir is not None else EVAL2_DIR
    return read_jsonl(d / QA_FILE), read_jsonl(d / CONTROL_FILE)
