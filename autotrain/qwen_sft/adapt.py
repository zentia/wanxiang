"""Answer-side teacher-forced metric on the frozen SFT holdout.

This is the RIGHT ruler for QLoRA adaptation: it measures whether the model
has learned the local code distribution on answer tokens. The generation
ruler (``ruler.py``) measures repetition, which is a different question --
using it alone to judge a style/domain adapter would be a category error.

All reports are sealed with ``adapt_hash`` (base + data corpus_id + n).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

VERSION = "wanxiang-qwen-adapt-1"
DEFAULT_N = 200


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score(
    base_dir: str | Path,
    data_dir: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    n: int = DEFAULT_N,
    load_in_4bit: bool = True,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    with (data / "holdout.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    if not rows:
        raise ValueError("no holdout rows")

    AutoTokenizer.from_pretrained(str(base_dir))  # validates tokenizer assets
    kw: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), **kw)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    total_nats = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for row in rows:
            ids = row["input_ids"]
            labels = row["labels"]
            x = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(input_ids=x).logits[0].float()   # (T, V)
            targets = torch.tensor(labels[1:], dtype=torch.long, device=logits.device)
            valid = targets != -100
            if not bool(valid.any()):
                continue
            pred = logits[:-1][valid]
            tgt = targets[valid]
            total_nats += float(F.cross_entropy(pred, tgt, reduction="sum").item())
            total_correct += int((pred.argmax(-1) == tgt).sum().item())
            total_tokens += int(tgt.numel())

    if total_tokens == 0:
        raise ValueError("no supervised answer tokens in holdout sample")
    spec = {
        "family": "wanxiang-qwen-adapt",
        "version": VERSION,
        "base": Path(base_dir).name,
        "data_corpus_id": manifest["corpus_id"],
        "n": len(rows),
    }
    return {
        "status": "OK",
        "version": VERSION,
        "adapt_hash": _sha(_canonical(spec))[:16],
        "base": Path(base_dir).name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "answer_cross_entropy": total_nats / total_tokens,
        "answer_top1_accuracy": total_correct / total_tokens,
        "answer_tokens": total_tokens,
        "samples": len(rows),
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before.get("adapt_hash") != after.get("adapt_hash"):
        return {
            "verdict": "INCOMPARABLE",
            "reason": "different adapt_hash (base / data / n differ)",
        }
    d_ce = after["answer_cross_entropy"] - before["answer_cross_entropy"]
    d_top1 = after["answer_top1_accuracy"] - before["answer_top1_accuracy"]
    verdict = (
        "IMPROVED" if d_ce < 0 and d_top1 > 0
        else "REGRESSED" if d_ce > 0 and d_top1 < 0
        else "NO_CHANGE"
    )
    return {
        "verdict": verdict,
        "adapt_hash": before["adapt_hash"],
        "answer_ce_delta": d_ce,
        "answer_top1_delta": d_top1,
        "before": {"ce": before["answer_cross_entropy"],
                   "top1": before["answer_top1_accuracy"]},
        "after": {"ce": after["answer_cross_entropy"],
                  "top1": after["answer_top1_accuracy"]},
    }
