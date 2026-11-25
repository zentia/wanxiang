"""Build QLoRA SFT data with the BASE model's own tokenizer.

The from-scratch SFT token streams (runs/code-sft-v2/*.bin) are encoded with
a 16384-word custom BPE and CANNOT be reused here. We re-encode from the
anonymised source jsonl, preserving:

* document-level frozen train/holdout split (no leakage)
* block-aligned, template-filtered, answer-deduplicated samples
* answer-side labels (-100 on prompt tokens)
* a sealed, path-free manifest
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..code_corpus.sft import _structural_samples
from ..corpus.pipeline import split_documents

VERSION = "wanxiang-qwen-sft-1"
INSTRUCTION = "complete the code below ({lang}):\n"
TRAIN_FILE = "train.jsonl"
HOLDOUT_FILE = "holdout.jsonl"
MANIFEST_FILE = "manifest.json"
_ABS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\[^\\]+\\")


class QwenSFTError(ValueError):
    """Corrupt or tampered SFT data."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_docs(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if len(docs) < 10:
        raise QwenSFTError("too few source documents")
    return docs


def build(
    base_dir: str | Path,
    source_jsonl: str | Path,
    out_dir: str | Path,
    *,
    max_samples: int = 20000,
    max_seq_len: int = 1024,
    holdout_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Write train/holdout jsonl (input_ids + labels) and a sealed manifest."""
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    if tok.eos_token_id is None:
        raise QwenSFTError("base tokenizer has no eos_token_id")

    docs = _read_docs(Path(source_jsonl))
    train_idx, holdout_idx = split_documents(
        len(docs), holdout_fraction=holdout_fraction, seed=seed
    )

    files: dict[str, Any] = {}
    counts: dict[str, int] = {}
    by_lang: Counter[str] = Counter()
    # Shared so a duplicated answer can never appear in both splits.
    seen_answers: set[str] = set()

    for split, indices in (("train", train_idx), ("holdout", holdout_idx)):
        # Deterministically shuffle the DOCUMENT order: truncating at
        # max_samples must not bias samples toward the first few documents.
        order = list(indices)
        random.Random(seed + (0 if split == "train" else 1)).shuffle(order)
        rows: list[dict[str, Any]] = []
        for i in order:
            if len(rows) >= max_samples:
                break
            d = docs[i]
            lang = d.get("lang", "code")
            lines = d["text"].split("\n")
            for prefix, answer in _structural_samples(lines, seen=seen_answers):
                if len(rows) >= max_samples:
                    break
                if len(prefix.strip()) < 20 or len(answer.strip()) < 20:
                    continue
                messages = [
                    {"role": "user", "content": INSTRUCTION.format(lang=lang) + prefix}
                ]
                prompt_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
                answer_ids = tok(answer + "\n", add_special_tokens=False)["input_ids"]
                answer_ids = list(answer_ids) + [int(tok.eos_token_id)]
                ids = list(prompt_ids) + answer_ids
                if len(ids) > max_seq_len or len(answer_ids) < 4:
                    continue
                labels = [-100] * len(prompt_ids) + answer_ids
                rows.append({"input_ids": ids, "labels": labels, "lang": lang})
                by_lang[lang] += 1

        if len(rows) < 5:
            raise QwenSFTError(f"{split} split produced only {len(rows)} samples")
        name = TRAIN_FILE if split == "train" else HOLDOUT_FILE
        p = out / name
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        files[name] = {"rows": len(rows), "sha256": _sha(p.read_text(encoding="utf-8"))}
        counts[split] = len(rows)

    doc: dict[str, Any] = {
        "version": VERSION,
        "base": Path(base_dir).name,
        "tokenizer_vocab_size": int(tok.vocab_size),
        "extraction": {
            "documents": len(docs),
            "train_documents": len(train_idx),
            "holdout_documents": len(holdout_idx),
            "samples": counts,
            "by_lang": dict(sorted(by_lang.items())),
            "max_seq_len": max_seq_len,
            "instruction": INSTRUCTION,
            "note": "block-aligned, answer-deduplicated; labels=-100 on prompt tokens",
        },
        "split": {
            "method": "seeded-document-index-permutation",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
        },
        "files": files,
    }
    doc["corpus_id"] = _sha(_canonical(doc))[:16]
    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify(out_dir: str | Path) -> dict[str, Any]:
    """Re-check seals, row counts and label invariants; reject path leaks."""
    out = Path(out_dir)
    doc = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    if doc.get("version") != VERSION:
        raise QwenSFTError("unsupported version")
    if _ABS_PATH_RE.search(json.dumps(doc, ensure_ascii=False)):
        raise QwenSFTError("manifest leaked an absolute filesystem path")
    for name, rec in doc["files"].items():
        p = out / name
        text = p.read_text(encoding="utf-8")
        if _sha(text) != rec["sha256"]:
            raise QwenSFTError(f"checksum mismatch: {name}")
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) != rec["rows"]:
            raise QwenSFTError(f"row count mismatch: {name}")
        for line in lines:
            row = json.loads(line)
            if len(row["input_ids"]) != len(row["labels"]):
                raise QwenSFTError("input_ids/labels length mismatch")
            if all(v == -100 for v in row["labels"]):
                raise QwenSFTError("sample has no supervised answer tokens")
    sealed = {k: v for k, v in doc.items() if k != "corpus_id"}
    if _sha(_canonical(sealed))[:16] != doc["corpus_id"]:
        raise QwenSFTError("corpus_id seal mismatch")
    return doc
