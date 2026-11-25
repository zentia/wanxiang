"""Frozen format / correctness proxy ruler for Qwen code adapters.

Closes the measurement gap found in CODE_LM.md 9.6: the adaptation ruler
measures distribution fit and the generation ruler measures repetition, but
neither detects "the model stopped chatting and started emitting code".

Two objective, judge-free metric groups on frozen SFT holdout prompts:

* FORMAT  - explanation_leak_rate (fixed, auditable phrase table),
            code_char_ratio / code_char_ratio_gap vs the reference,
            indented_line_ratio
* QUALITY - raw_char_f1 (punctuation-preserving), plus the project's existing
            eval2 score_pair (char_f1 / rouge_l / edit_similarity /
            exact_match / composite)

``quality_hash`` binds the ruler only (data corpus_id, base, n, gen_len,
phrase table). Decode controls are the variable under test, not part of it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ..eval2.metrics import score_pair

VERSION = "wanxiang-qwen-quality-1"
N_SAMPLES = 32
GEN_LEN = 64
SEED = 0

# Fixed, auditable markers of an explanatory (non-completion) reply. No LLM,
# no learned judge: a deterministic substring test on the first 80 chars.
EXPLANATION_MARKERS = (
    "certainly", "sure,", "sure!", "here's", "here is", "it looks like",
    "you're", "you are", "this code", "the code", "i can", "i'll",
    "to complete", "in this example", "below is", "the following",
    "as an ai", "当然", "这是", "下面是", "你可以", "这段代码", "以下", "我来",
)
# Structural characters (ordinary spaces excluded on purpose).
CODE_CHARS = set("{}();=,.<>[]:/+*-&|!?\"'\n\t")

_HIGHER_BETTER = (
    "code_char_ratio", "indented_line_ratio", "raw_char_f1",
    "char_f1", "rouge_l", "edit_similarity", "exact_match", "composite",
)
_LOWER_BETTER = ("explanation_leak_rate", "code_char_ratio_gap")


class QualityError(ValueError):
    """Frozen ruler violation."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pure metrics (unit-testable without a model).
# ---------------------------------------------------------------------------


def explanation_leak(text: str) -> float:
    head = text[:80].lower()
    return 1.0 if any(m in head for m in EXPLANATION_MARKERS) else 0.0


def code_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in CODE_CHARS) / len(text)


def indented_line_ratio(text: str) -> float:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if l[0] in " \t") / len(lines)


def raw_char_f1(pred: str, ref: str) -> float:
    """Bag-of-characters F1 WITHOUT dropping punctuation/whitespace."""
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    cp, cr = Counter(pred), Counter(ref)
    overlap = sum(min(cp[c], cr[c]) for c in cp.keys() & cr.keys())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def pair_metrics(gen_text: str, ref_text: str) -> dict[str, float]:
    """All per-sample metrics; pure function."""
    out = {
        "explanation_leak_rate": explanation_leak(gen_text),
        "code_char_ratio": code_char_ratio(gen_text),
        "code_char_ratio_gap": abs(code_char_ratio(gen_text) - code_char_ratio(ref_text)),
        "indented_line_ratio": indented_line_ratio(gen_text),
        "raw_char_f1": raw_char_f1(gen_text, ref_text),
    }
    out.update(score_pair(gen_text, ref_text))
    return out


# ---------------------------------------------------------------------------
# Frozen samples.
# ---------------------------------------------------------------------------


def freeze_samples(data_dir: str | Path, *, n: int = N_SAMPLES, seed: int = SEED):
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    with (data / "holdout.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < n:
        raise QualityError("not enough holdout rows")
    step = len(rows) // n
    picks = [rows[i * step] for i in range(n)]
    samples = []
    for r in picks:
        prompt_len = next((i for i, v in enumerate(r["labels"]) if v != -100),
                          len(r["labels"]))
        prompt_ids = r["input_ids"][:prompt_len]
        ref_ids = [t for t, v in zip(r["input_ids"], r["labels"]) if v != -100]
        if prompt_ids and ref_ids:
            samples.append({"prompt_ids": prompt_ids, "ref_ids": ref_ids})
    payload = {
        "samples": samples,
        "corpus_id": manifest["corpus_id"],
        "n": len(samples),
        "markers": list(EXPLANATION_MARKERS),
    }
    return {
        "samples": samples,
        "sha256": _sha(_canonical(payload)),
        "corpus_id": manifest["corpus_id"],
    }


def build_spec(sample_sha: str, corpus_id: str, base: str, *, n: int,
               gen_len: int) -> dict[str, Any]:
    doc = {
        "family": "wanxiang-qwen-quality",
        "version": VERSION,
        "sample_sha256": sample_sha,
        "corpus_id": corpus_id,
        "base": base,
        "n_samples": n,
        "gen_len": gen_len,
        "seed": SEED,
        "markers_sha256": _sha(_canonical(list(EXPLANATION_MARKERS)))[:16],
        "metric": "objective format + correctness-proxy stats on frozen prompts",
    }
    doc["quality_hash"] = _sha(_canonical(doc))[:16]
    return doc


# ---------------------------------------------------------------------------
# Score / compare.
# ---------------------------------------------------------------------------


def score(
    base_dir: str | Path,
    data_dir: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    decode: dict[str, Any] | None = None,
    n: int = N_SAMPLES,
    gen_len: int = GEN_LEN,
    load_in_4bit: bool = True,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    decode = dict(decode or {})
    frozen = freeze_samples(data_dir, n=n)
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
    examples: list[dict[str, str]] = []
    for s in frozen["samples"]:
        x = torch.tensor([s["prompt_ids"]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=x, max_new_tokens=gen_len, do_sample=False,
                pad_token_id=pad_id, **decode,
            )
        gen_ids = [int(t) for t in out[0][x.shape[1]:].tolist()]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        ref_text = tok.decode(s["ref_ids"], skip_special_tokens=True)
        rows.append(pair_metrics(gen_text, ref_text))
        if len(examples) < 2:
            examples.append({"generated": gen_text, "reference": ref_text})

    spec = build_spec(frozen["sha256"], frozen["corpus_id"],
                      Path(base_dir).name, n=len(rows), gen_len=gen_len)
    return {
        "status": "OK",
        "version": VERSION,
        "quality_hash": spec["quality_hash"],
        "spec": spec,
        "base": Path(base_dir).name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "decode_config": {
            "repetition_penalty": decode.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": decode.get("no_repeat_ngram_size", 0),
            "do_sample": False,
        },
        "metrics": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
        "n_samples": len(rows),
        "examples": examples,
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before.get("quality_hash") != after.get("quality_hash"):
        return {
            "verdict": "INCOMPARABLE",
            "reason": "different quality_hash (data / base / n / markers differ)",
        }
    b, a = before["metrics"], after["metrics"]
    deltas = {k: a[k] - b[k] for k in _HIGHER_BETTER + _LOWER_BETTER}
    improved = sum(deltas[k] > 0 if k in _HIGHER_BETTER else deltas[k] < 0
                   for k in deltas)
    regressed = sum(deltas[k] < 0 if k in _HIGHER_BETTER else deltas[k] > 0
                    for k in deltas)
    verdict = ("IMPROVED" if improved > regressed
               else "REGRESSED" if regressed > improved else "NO_CHANGE")
    return {
        "verdict": verdict,
        "quality_hash": before["quality_hash"],
        "metrics_before": b,
        "metrics_after": a,
        "deltas": deltas,
        "improved_metrics": improved,
        "regressed_metrics": regressed,
    }
