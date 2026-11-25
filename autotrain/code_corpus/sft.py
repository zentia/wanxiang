"""Code SFT: instruction-formatted completion pairs + a frozen objective ruler.

Data is derived ONLY from the anonymised code jsonl. Each sample is a
single-turn conversation: the user pastes a code prefix and asks for a
completion; the assistant reply is the following lines. Labels mark just the
reply tokens so loss is answer-side (``TokenDataset(label_file=...)``).

The holdout uses the SAME document split as the pretraining corpus, so no code
that was evaluated on is ever trained on. ``score`` reports teacher-forced
answer-side cross-entropy / top-1 accuracy -- objective, no LLM-as-judge.

All artifacts are local derived data under ``runs/`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..corpus.pipeline import split_documents
from ..eval_lm.metrics import cross_entropy_from_logits, perplexity
from ..export.loader import load_bundle
from ..inference.chat import TEMPLATE_VERSION, ChatTemplate
from ..inference.tokenizer import EOS_ID, load_tokenizer

SFT_VERSION = "wanxiang-code-sft-1"
TOKENS_TRAIN = "sft_train_tokens.bin"
LABELS_TRAIN = "sft_train_labels.bin"
TOKENS_HOLDOUT = "sft_holdout_tokens.bin"
LABELS_HOLDOUT = "sft_holdout_labels.bin"
TEMPLATE_FILE = "chat_template.json"
MANIFEST_FILE = "manifest.json"

# ASCII markers: every character here is guaranteed to occur in a source-code
# BPE vocab (unlike the Chinese idiom template). Name must stay TEMPLATE_VERSION
# so the inference server's ChatTemplate.from_file() accepts it.
CODE_CHAT_SPEC: dict[str, Any] = {
    "name": TEMPLATE_VERSION,
    "role_prefix": {
        "system": "sys:",
        "user": "user:",
        "assistant": "code:",
    },
    "turn_sep": "\n",
    "generation_prefix": "code:",
    "note": "ASCII markers for the code BPE vocab; special tokens are ids.",
}
INSTRUCTION = "complete the code below ({lang}):\n"

# Manifests must stay free of absolute filesystem paths (anonymised corpora
# are derived from a private local source tree whose location is never stored).
_ABS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\[^\\]+\\")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sample generators.
# ---------------------------------------------------------------------------

_BLOCK_MIN_LINES = 5
_BLOCK_MAX_LINES = 60
_STRUCT_MAX_ANSWER_LINES = 10


def _find_blocks(lines: list[str]):
    """Yield (start, end) for balanced brace-delimited blocks.

    Block-aligned cuts never slice through the middle of a function the way a
    fixed line window does. Includes control blocks (if/for/while), which are
    legitimate short completion targets.
    """
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if "{" in line and line.strip():
            depth = line.count("{") - line.count("}")
            if depth > 0:
                j = i + 1
                while j < n and depth > 0:
                    depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                if depth == 0 and _BLOCK_MIN_LINES <= j - i <= _BLOCK_MAX_LINES:
                    yield i, j
                    i = j
                    continue
        i += 1


def _is_template(body: list[str]) -> bool:
    """Reject boilerplate that teaches the model to repeat itself."""
    stripped = [l.strip() for l in body if l.strip()]
    if len(stripped) < _BLOCK_MIN_LINES:
        return True
    code = [l for l in stripped
            if not l.startswith(("//", "///", "#", "/*", "*", "--"))]
    if len(code) < 3:
        return True
    if len(set(stripped)) < len(stripped) * 0.6:
        return True
    if re.search(r"\{\s*get;\s*set;\s*\}", "\n".join(body)):
        return True
    return False


def _window_samples(lines: list[str], prefix_lines: int,
                    answer_lines: int, stride: int):
    """Legacy fixed-line sliding window (kept for reproducibility)."""
    for start in range(0, max(0, len(lines) - prefix_lines - 2), stride):
        yield (
            "\n".join(lines[start:start + prefix_lines]),
            "\n".join(lines[start + prefix_lines:start + prefix_lines + answer_lines]),
        )


def _structural_samples(lines: list[str], seen: set[str] | None = None):
    """Block-aligned, template-filtered, answer-deduplicated samples."""
    for start, end in _find_blocks(lines):
        body = lines[start:end]
        if _is_template(body):
            continue
        size = len(body)
        prompt_len = min(max(3, size // 3), 12)
        answer_len = min(max(4, size - prompt_len), _STRUCT_MAX_ANSWER_LINES)
        if answer_len < 3:
            continue
        prefix = "\n".join(body[:prompt_len])
        answer = "\n".join(body[prompt_len:prompt_len + answer_len])
        if seen is not None:
            key = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
        yield prefix, answer


def build_sft(
    source_jsonl: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    *,
    holdout_fraction: float = 0.05,
    seed: int = 42,
    prefix_lines: int = 8,
    answer_lines: int = 12,
    stride: int = 32,
    max_seq_len: int = 512,
    max_samples: int = 200000,
    mode: str = "window",
) -> dict[str, Any]:
    """Write train/holdout SFT token+label streams, template and sealed manifest."""
    src = Path(source_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(tokenizer_path)
    if not getattr(tok, "supports_text", False):
        raise ValueError("SFT requires a text-capable tokenizer")
    tmpl = ChatTemplate(CODE_CHAT_SPEC)
    tmpl.validate_chars(tok)

    docs = _read_jsonl(src)
    train_idx, holdout_idx = split_documents(
        len(docs), holdout_fraction=holdout_fraction, seed=seed
    )

    instruction_chars = set(INSTRUCTION.replace("{lang}", ""))
    missing = sorted(ch for ch in instruction_chars if ch not in tok.token_to_id)
    if missing:
        raise ValueError(f"instruction characters missing from vocab: {missing!r}")

    tmpl.save(out / TEMPLATE_FILE)
    # Self-contained: the SFT directory carries its own tokenizer so a Mac
    # sync needs only this one folder, and verify can re-check its checksum.
    shutil.copy2(Path(tokenizer_path), out / "tokenizer.json")

    files: dict[str, Any] = {}
    sample_counts: dict[str, int] = {}
    by_lang: Counter[str] = Counter()

    # Shared across splits so a template answer can never leak into holdout.
    seen_answers: set[str] = set()
    for split, indices in (("train", train_idx), ("holdout", holdout_idx)):
        tokens: list[int] = []
        labels: list[int] = []
        n_samples = 0
        for i in indices:
            if n_samples >= max_samples:
                break
            d = docs[i]
            lang = d.get("lang", "code")
            lines = d["text"].split("\n")
            if mode == "block":
                candidates = _structural_samples(lines, seen=seen_answers)
            elif mode == "window":
                candidates = _window_samples(lines, prefix_lines, answer_lines, stride)
            else:
                raise ValueError(f"unknown sft mode: {mode!r}")
            for prefix, answer in candidates:
                if n_samples >= max_samples:
                    break
                if len(prefix.strip()) < 20 or len(answer.strip()) < 20:
                    continue

                prompt_ids = tmpl.encode_messages(
                    [{"role": "user", "content": INSTRUCTION.format(lang=lang) + prefix}],
                    tok, add_generation_prompt=True,
                )
                answer_ids = tok.encode(answer + "\n") + [int(EOS_ID)]
                ids = prompt_ids + answer_ids
                if len(ids) > max_seq_len:
                    # Keep the whole prompt, trim the reply to fit.
                    room = max_seq_len - len(prompt_ids)
                    if room < 16:
                        continue
                    answer_ids = answer_ids[:room - 1] + [int(EOS_ID)]
                    ids = prompt_ids + answer_ids
                if len(set(ids) - set(range(tok.vocab_size))):
                    continue

                lab = [0] * len(prompt_ids) + [1] * len(answer_ids)
                tokens.extend(ids)
                labels.extend(lab)
                n_samples += 1
                by_lang[lang] += 1

        if n_samples < 5:
            raise ValueError(f"{split} split produced only {n_samples} SFT samples")

        t_file = TOKENS_TRAIN if split == "train" else TOKENS_HOLDOUT
        l_file = LABELS_TRAIN if split == "train" else LABELS_HOLDOUT
        (out / t_file).write_bytes(np.asarray(tokens, dtype="<u2").tobytes())
        (out / l_file).write_bytes(np.asarray(labels, dtype=np.uint8).tobytes())
        for name in (t_file, l_file):
            p = out / name
            files[name] = {"sha256": _sha256_bytes(p), "bytes": p.stat().st_size}
        files[t_file]["tokens"] = len(tokens)
        files[l_file]["labels"] = len(labels)
        sample_counts[split] = n_samples

    tok_copy = out / "tokenizer.json"
    files["tokenizer.json"] = {
        "sha256": _sha256_bytes(tok_copy),
        "bytes": tok_copy.stat().st_size,
    }

    doc: dict[str, Any] = {
        "version": SFT_VERSION,
        "source": {
            "name": "local-code-sft",
            "license": "proprietary-local",
            "description": "anonymised instruction pairs derived from the "
                           "frozen local code corpus; no paths recorded",
        },
        "extraction": {
            "documents": len(docs),
            "train_documents": len(train_idx),
            "holdout_documents": len(holdout_idx),
            "samples": sample_counts,
            "by_lang": dict(sorted(by_lang.items())),
            "prefix_lines": prefix_lines,
            "answer_lines": answer_lines,
            "stride": stride,
            "max_seq_len": max_seq_len,
            "mode": mode,
            "answer_dedup": mode == "block",
            "instruction": INSTRUCTION,
            "note": "single-turn code completion; answer-side labels (uint8)",
        },
        "split": {
            "method": "seeded-document-index-permutation (same as pretrain corpus)",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "note": "SFT holdout comes ONLY from pretraining holdout documents",
        },
        "tokenizer": {
            "sha256": tok.as_dict()["sha256"],
            "vocab_size": tok.vocab_size,
            "type": tok.type,
        },
        "template": {
            "hash": tmpl.template_hash,
            "name": TEMPLATE_VERSION,
        },
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    doc["corpus_id"] = hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16]
    (out / MANIFEST_FILE).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return doc


def verify_sft(out_dir: str | Path) -> dict[str, Any]:
    """Re-check every SFT artifact and the seal; reject path leaks."""
    out = Path(out_dir)
    p = out / MANIFEST_FILE
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("version") != SFT_VERSION:
        raise ValueError("unsupported SFT corpus version")
    for name, rec in doc["files"].items():
        f = out / name
        if not f.is_file() or _sha256_bytes(f) != rec["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
        if f.stat().st_size != rec["bytes"]:
            raise ValueError(f"size mismatch: {name}")
    for tok_f, lab_f in ((TOKENS_TRAIN, LABELS_TRAIN),
                         (TOKENS_HOLDOUT, LABELS_HOLDOUT)):
        if (out / tok_f).stat().st_size // 2 != (out / lab_f).stat().st_size:
            raise ValueError("token/label length mismatch")
    tmpl = ChatTemplate.from_file(out / TEMPLATE_FILE)
    if tmpl.template_hash != doc["template"]["hash"]:
        raise ValueError("chat template seal mismatch")
    tok = load_tokenizer(out / "tokenizer.json")
    if tok.as_dict()["sha256"] != doc["tokenizer"]["sha256"]:
        raise ValueError("tokenizer checksum mismatch")
    if tok.vocab_size != doc["tokenizer"]["vocab_size"]:
        raise ValueError("tokenizer vocab_size mismatch")
    sealed = {k: v for k, v in doc.items() if k not in ("corpus_id", "generated_at")}
    if hashlib.sha256(_canonical(sealed).encode("utf-8")).hexdigest()[:16] != doc["corpus_id"]:
        raise ValueError("corpus_id seal mismatch")
    blob = json.dumps(doc, ensure_ascii=False)
    if _ABS_PATH_RE.search(blob):
        raise ValueError("manifest leaked an absolute filesystem path")
    return doc


# ---------------------------------------------------------------------------
# Objective ruler: teacher-forced answer-side CE + top-1 accuracy.
# ---------------------------------------------------------------------------


def score_sft(
    bundle_dir: str | Path,
    corpus_dir: str | Path,
    *,
    seq_len: int = 512,
    batch_size: int = 8,
    device: str = "cpu",
    max_windows: int = 400,
) -> dict[str, Any]:
    """Score answer tokens only on the frozen SFT holdout."""
    out = Path(corpus_dir)
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    tokens = np.fromfile(out / TOKENS_HOLDOUT, dtype="<u2").astype(np.int64)
    labels = np.fromfile(out / LABELS_HOLDOUT, dtype=np.uint8)

    model, info = load_bundle(bundle_dir, device=device)
    model.eval()

    total_nats = 0.0
    total_valid = 0
    total_correct = 0
    windows = 0
    with torch.no_grad():
        for start in range(0, min(len(tokens) - seq_len - 1, max_windows * seq_len * batch_size),
                           seq_len * batch_size):
            batch_x: list[np.ndarray] = []
            batch_y: list[np.ndarray] = []
            for b in range(batch_size):
                s = start + b * seq_len
                if s + seq_len + 1 > len(tokens):
                    break
                x = tokens[s:s + seq_len]
                y = tokens[s + 1:s + seq_len + 1].copy()
                lab = labels[s + 1:s + seq_len + 1]
                y[lab == 0] = -1
                batch_x.append(x)
                batch_y.append(y)
            if not batch_x:
                break
            xb = torch.tensor(np.stack(batch_x), dtype=torch.long, device=device)
            yb = torch.tensor(np.stack(batch_y), dtype=torch.long, device=device)
            logits, _ = model(xb)
            total_nats += cross_entropy_from_logits(logits, yb) * int((yb != -1).sum())
            valid = yb != -1
            total_correct += int(((logits.argmax(-1) == yb) & valid).sum())
            total_valid += int(valid.sum())
            windows += len(batch_x)

    if total_valid == 0:
        raise ValueError("no answer tokens in SFT holdout")
    ce = total_nats / total_valid
    spec_src = {
        "corpus_id": manifest["corpus_id"],
        "template_hash": manifest["template"]["hash"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    return {
        "status": "OK",
        "sft_hash": hashlib.sha256(_canonical(spec_src).encode("utf-8")).hexdigest()[:16],
        "bundle_step": info.step,
        "answer_cross_entropy": ce,
        "answer_perplexity": perplexity(ce),
        "answer_top1_accuracy": total_correct / total_valid,
        "answer_tokens": total_valid,
        "windows": windows,
        "note": "Teacher-forced answer-side CE on frozen SFT holdout. "
                "NOT comparable with eval_lm perplexity (different ruler).",
    }
