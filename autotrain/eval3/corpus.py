"""Build and byte-verify the frozen wanxiang-eval-3 word-holdout corpus.

Holdout is by IDIOM WORD (see autotrain.split). Those words are excluded from
both pretrain and SFT streams, so every question here is genuinely unseen by
the model. The reference answer is produced by the SAME clean_answer() used
to build the SFT labels -- train and eval are byte-identical by construction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..inference.chat import ChatTemplate
from ..inference.tokenizer import Tokenizer
from ..sft.corpus import QUESTION_TEMPLATES
from ..split import (HOLDOUT_COUNT, HOLDOUT_SEED, answer_for_row,
                     holdout_words, split_rows)
from .spec import (CONTROL_ROTATION, EVAL3_VERSION, FROZEN, QA_COUNT,
                   _canonical)

EVAL3_DIR = Path(__file__).resolve().parents[2] / "evals3"
QA_FILE = "qa.jsonl"
CONTROL_FILE = "negative_control.jsonl"
WORDS_FILE = "holdout_words.json"
MANIFEST_FILE = "manifest.json"
EXPECTED_FILES = (QA_FILE, CONTROL_FILE, WORDS_FILE, MANIFEST_FILE)


class Eval3CorpusError(ValueError):
    """Frozen eval-3 corpus missing, malformed or tampered with."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )


def freeze(
    source_json: str | Path,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    out_dir: str | Path | None = None,
    *,
    count: int = QA_COUNT,
    seed: int = HOLDOUT_SEED,
    holdout_n: int = HOLDOUT_COUNT,
) -> dict[str, Any]:
    """Create the word-holdout QA set + shuffled control + sealed manifest."""
    out = Path(out_dir) if out_dir is not None else EVAL3_DIR
    out.mkdir(parents=True, exist_ok=True)
    chat_template.validate_chars(tokenizer)

    rows = json.loads(Path(source_json).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise Eval3CorpusError("source JSON must be an array")

    holdout = holdout_words(rows, seed=seed, count=holdout_n)
    _, held_rows = split_rows(rows, set(holdout))

    samples: list[dict[str, Any]] = []
    for row in held_rows:
        word = str(row.get("word", "")).strip()
        answer = answer_for_row(row)
        if not word or not answer:
            continue
        question = QUESTION_TEMPLATES[len(samples) % len(QUESTION_TEMPLATES)].format(word=word)
        samples.append({"word": word, "question": question, "reference": answer})
        if len(samples) >= count:
            break
    if len(samples) < count:
        raise Eval3CorpusError(
            f"only {len(samples)} usable holdout samples, need {count}"
        )

    control = [
        {
            "question": samples[i]["question"],
            "reference": samples[(i + CONTROL_ROTATION) % count]["reference"],
            "note": "shuffled reference: same question, wrong answer",
        }
        for i in range(count)
    ]

    _write_jsonl(out / QA_FILE, samples)
    _write_jsonl(out / CONTROL_FILE, control)
    (out / WORDS_FILE).write_text(
        json.dumps({"seed": seed, "count": holdout_n, "words": holdout},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    doc: dict[str, Any] = {
        "format_version": EVAL3_VERSION,
        "frozen": FROZEN,
        "qa_count": count,
        "holdout_seed": seed,
        "holdout_count": holdout_n,
        "control_rotation": CONTROL_ROTATION,
        "holdout_strategy": "word-level: all samples of a holdout word excluded "
                            "from pretrain and SFT streams",
        "answer_source": "autotrain.split.answer_for_row (shared with SFT labels)",
        "source": {
            "path": str(source_json),
            "sha256": _sha256_file(Path(source_json)),
        },
        "tokenizer": {"type": tokenizer.type, "vocab_size": tokenizer.vocab_size,
                      "sha256": tokenizer.as_dict()["sha256"]},
        "chat_template": {"version": chat_template.name,
                          "template_hash": chat_template.template_hash},
        "files": {
            name: {"sha256": _sha256_file(out / name),
                   "bytes": (out / name).stat().st_size,
                   "rows": count if name in (QA_FILE, CONTROL_FILE) else None}
            for name in (QA_FILE, CONTROL_FILE, WORDS_FILE)
        },
        "note": "Frozen word-holdout instrument. Never regenerate in place. "
                "INCOMPARABLE with eval-2 by design.",
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
    d = Path(eval_dir) if eval_dir is not None else EVAL3_DIR
    if not d.is_dir():
        raise Eval3CorpusError(f"eval-3 corpus directory not found: {d}")
    actual = sorted(p.name for p in d.iterdir() if p.is_file())
    if actual != sorted(EXPECTED_FILES):
        raise Eval3CorpusError(
            f"eval-3 file set mismatch: expected {sorted(EXPECTED_FILES)}, got {actual}"
        )
    p = d / MANIFEST_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise Eval3CorpusError(f"invalid manifest JSON: {e}") from e
    if doc.get("format_version") != EVAL3_VERSION or doc.get("frozen") is not True:
        raise Eval3CorpusError("eval-3 corpus is not a frozen wanxiang-eval-3 set")

    for name, rec in doc["files"].items():
        f = d / name
        if _sha256_file(f) != rec["sha256"]:
            raise Eval3CorpusError(f"checksum mismatch: {name}")
        if f.stat().st_size != rec["bytes"]:
            raise Eval3CorpusError(f"size mismatch: {name}")
        if rec.get("rows") is not None and len(read_jsonl(f)) != rec["rows"]:
            raise Eval3CorpusError(f"row count mismatch: {name}")

    sealed = {k: v for k, v in doc.items() if k != "corpus_manifest_hash"}
    expected = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()
    if expected != doc.get("corpus_manifest_hash"):
        raise Eval3CorpusError("corpus_manifest_hash seal mismatch (corpus edited)")
    return doc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_qa(eval_dir: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d = Path(eval_dir) if eval_dir is not None else EVAL3_DIR
    return read_jsonl(d / QA_FILE), read_jsonl(d / CONTROL_FILE)
