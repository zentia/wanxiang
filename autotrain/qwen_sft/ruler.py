"""Frozen objective generation ruler for the Qwen-base family.

Reuses the pure metrics/compare functions from ``autotrain.eval_gen`` verbatim;
only the frozen prompt set is rebuilt with the BASE model's tokenizer, because
the from-scratch prompt ids belong to a completely different vocabulary.

``gen_hash`` binds the ruler (prompt set, lengths, seed, base name). Decode
controls are NOT part of the hash: they are the variable under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..corpus.pipeline import split_documents
from ..eval_gen.runner import compare, sequence_metrics

VERSION = "wanxiang-qwen-gen-1"
N_PROMPTS = 16
PROMPT_TOKENS = 64
GEN_LEN = 64
SEED = 0


class QwenGenError(ValueError):
    """Frozen ruler violation."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_prompts(
    base_dir: str | Path,
    source_jsonl: str | Path,
    *,
    n: int = N_PROMPTS,
    prompt_tokens: int = PROMPT_TOKENS,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic prompts from the frozen (document-level) holdout docs."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(base_dir))
    docs = []
    with Path(source_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    _train, holdout = split_documents(len(docs), holdout_fraction=0.05, seed=seed)
    if len(holdout) < n:
        raise QwenGenError("not enough holdout documents for the prompt set")
    picks = [holdout[int(i * len(holdout) / n)] for i in range(n)]
    prompts: list[list[int]] = []
    for i in picks:
        ids = tok(docs[i]["text"], add_special_tokens=False)["input_ids"][:prompt_tokens]
        if len(ids) >= 8:
            prompts.append([int(x) for x in ids])
    payload = {
        "prompts": prompts, "base": Path(base_dir).name,
        "prompt_tokens": prompt_tokens, "n": len(prompts),
    }
    return {
        "prompts": prompts,
        "sha256": _sha(_canonical(payload)),
        "base": Path(base_dir).name,
    }


def build_spec(prompt_sha: str, base: str, *, n: int, gen_len: int) -> dict[str, Any]:
    doc = {
        "family": "wanxiang-qwen-gen",
        "version": VERSION,
        "prompt_sha256": prompt_sha,
        "base": base,
        "n_prompts": n,
        "prompt_tokens": PROMPT_TOKENS,
        "gen_len": gen_len,
        "seed": SEED,
        "decode": "greedy (do_sample=False)",
        "metric": "objective repetition stats on frozen holdout prompts",
    }
    doc["gen_hash"] = _sha(_canonical(doc))[:16]
    return doc


def score(
    base_dir: str | Path,
    source_jsonl: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    decode: dict[str, Any] | None = None,
    n_prompts: int = N_PROMPTS,
    gen_len: int = GEN_LEN,
    load_in_4bit: bool = True,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    decode = dict(decode or {})
    prompts = freeze_prompts(base_dir, source_jsonl, n=n_prompts)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
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
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows: list[dict[str, float]] = []
    for p in prompts["prompts"]:
        x = torch.tensor([p], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=x,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=pad_id,
                **decode,
            )
        generated = [int(t) for t in out[0][x.shape[1]:].tolist()]
        rows.append(sequence_metrics(generated))

    spec = build_spec(prompts["sha256"], prompts["base"],
                      n=len(prompts["prompts"]), gen_len=gen_len)
    return {
        "status": "OK",
        "version": VERSION,
        "gen_hash": spec["gen_hash"],
        "spec": spec,
        "base": prompts["base"],
        "adapter": str(adapter_dir) if adapter_dir else None,
        "decode_config": {
            "repetition_penalty": decode.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": decode.get("no_repeat_ngram_size", 0),
            "do_sample": False,
        },
        "metrics": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
        "n_prompts": len(rows),
    }


__all__ = ["score", "compare", "freeze_prompts", "build_spec", "VERSION"]
