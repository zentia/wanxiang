"""Frozen SYNTAX ruler -- real parser diagnostics, not a character proxy.

Uses Roslyn (shipped with the local dotnet SDK) via a tiny C# helper to parse
each generated C# fragment and count syntax-error diagnostics. For all
languages it also measures structural balance vs the reference continuation.

This is the fourth ruler: adapt (distribution), ruler (repetition), quality
(format / instruction following), syntax (real grammar). No LLM-as-judge.

``syntax_hash`` binds data corpus_id / base / n / gen_len / wrapper version.
Decode controls remain the variable under test, outside the hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

VERSION = "wanxiang-qwen-syntax-1"
WRAPPER_VERSION = "class __W { void __M() { <fragment> } }"
N_SAMPLES = 32
GEN_LEN = 64
SEED = 0
DEFAULT_CHECKER = "runs/syntaxcheck/bin/SyntaxCheck.exe"

_HIGHER_BETTER = ("syntax_error_free_rate",)
_LOWER_BETTER = (
    "avg_syntax_errors", "brace_balance_gap", "paren_balance_gap",
    "bracket_balance_gap", "unterminated_quote_rate",
)


class SyntaxRulerError(ValueError):
    """Frozen ruler or checker problem."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pure structural helpers (unit-testable).
# ---------------------------------------------------------------------------


def balance(text: str, opening: str, closing: str) -> int:
    return text.count(opening) - text.count(closing)


def unterminated_quotes(text: str) -> float:
    # Rough but deterministic: an odd number of unescaped double quotes.
    n = text.count('"') - text.count('\\"')
    return 1.0 if n % 2 else 0.0


def structure_metrics(gen: str, ref: str) -> dict[str, float]:
    return {
        "brace_balance_gap": abs(balance(gen, "{", "}") - balance(ref, "{", "}")),
        "paren_balance_gap": abs(balance(gen, "(", ")") - balance(ref, "(", ")")),
        "bracket_balance_gap": abs(balance(gen, "[", "]") - balance(ref, "[", "]")),
        "unterminated_quote_rate": unterminated_quotes(gen),
    }


# ---------------------------------------------------------------------------
# Roslyn checker helper.
# ---------------------------------------------------------------------------


def ensure_checker(checker: str | Path | None = None) -> Path:
    p = Path(checker or DEFAULT_CHECKER)
    if p.is_file():
        return p
    src = Path(__file__).resolve().parent / "roslyn" / "SyntaxCheck.csproj"
    if not src.is_file():
        raise SyntaxRulerError(f"checker missing and source project not found: {src}")
    p.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["dotnet", "build", str(src), "-c", "Release", "-o", str(p.parent),
         "--nologo"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not p.is_file():
        raise SyntaxRulerError(
            f"failed to build Roslyn checker ({p}). dotnet output:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
        )
    return p


def syntax_errors(fragments: list[str], checker: Path) -> list[int]:
    """One subprocess call for the whole batch; JSON string per stdin line."""
    payload = "\n".join(json.dumps(f, ensure_ascii=False) for f in fragments) + "\n"
    proc = subprocess.run(
        [str(checker)], input=payload, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SyntaxRulerError(f"checker failed: {proc.stderr[-500:]}")
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if len(lines) != len(fragments):
        raise SyntaxRulerError(
            f"checker returned {len(lines)} rows for {len(fragments)} fragments"
        )
    out = []
    for l in lines:
        v = json.loads(l)["errors"]
        if v < 0:
            raise SyntaxRulerError("checker could not decode a fragment")
        out.append(int(v))
    return out


# ---------------------------------------------------------------------------
# Frozen samples / spec.
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
        raise SyntaxRulerError("not enough holdout rows")
    step = len(rows) // n
    picks = [rows[i * step] for i in range(n)]
    samples = []
    for r in picks:
        prompt_len = next((i for i, v in enumerate(r["labels"]) if v != -100),
                          len(r["labels"]))
        prompt_ids = r["input_ids"][:prompt_len]
        ref_ids = [t for t, v in zip(r["input_ids"], r["labels"]) if v != -100]
        if prompt_ids and ref_ids:
            samples.append({
                "prompt_ids": prompt_ids,
                "ref_ids": ref_ids,
                "lang": r.get("lang", "unknown"),
            })
    payload = {"samples": samples, "corpus_id": manifest["corpus_id"],
               "n": len(samples)}
    return {
        "samples": samples,
        "sha256": _sha(_canonical(payload)),
        "corpus_id": manifest["corpus_id"],
    }


def build_spec(sample_sha: str, corpus_id: str, base: str, *, n: int,
               gen_len: int) -> dict[str, Any]:
    doc = {
        "family": "wanxiang-qwen-syntax",
        "version": VERSION,
        "sample_sha256": sample_sha,
        "corpus_id": corpus_id,
        "base": base,
        "n_samples": n,
        "gen_len": gen_len,
        "seed": SEED,
        "wrapper": WRAPPER_VERSION,
        "parser": "Roslyn CSharpSyntaxTree.ParseText error diagnostics",
    }
    doc["syntax_hash"] = _sha(_canonical(doc))[:16]
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
    checker: str | Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    decode = dict(decode or {})
    frozen = freeze_samples(data_dir, n=n)
    tool = ensure_checker(checker)
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

    gen_texts: list[str] = []
    ref_texts: list[str] = []
    cs_indices: list[int] = []
    for idx, s in enumerate(frozen["samples"]):
        x = torch.tensor([s["prompt_ids"]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=x, max_new_tokens=gen_len, do_sample=False,
                pad_token_id=pad_id, **decode,
            )
        gen_texts.append(tok.decode([int(t) for t in out[0][x.shape[1]:].tolist()],
                                   skip_special_tokens=True))
        ref_texts.append(tok.decode(s["ref_ids"], skip_special_tokens=True))
        if s["lang"] == "cs":
            cs_indices.append(idx)

    if not cs_indices:
        raise SyntaxRulerError("no C# samples in the frozen prompt set")
    errors = syntax_errors([gen_texts[i] for i in cs_indices], tool)

    struct = [structure_metrics(g, r) for g, r in zip(gen_texts, ref_texts)]
    metrics = {
        "syntax_error_free_rate": sum(1 for e in errors if e == 0) / len(errors),
        "avg_syntax_errors": sum(errors) / len(errors),
        "cs_samples": len(cs_indices),
        **{k: sum(m[k] for m in struct) / len(struct) for k in struct[0]},
    }

    spec = build_spec(frozen["sha256"], frozen["corpus_id"],
                      Path(base_dir).name, n=len(gen_texts), gen_len=gen_len)
    return {
        "status": "OK",
        "version": VERSION,
        "syntax_hash": spec["syntax_hash"],
        "spec": spec,
        "base": Path(base_dir).name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "decode_config": {
            "repetition_penalty": decode.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": decode.get("no_repeat_ngram_size", 0),
            "do_sample": False,
        },
        "metrics": metrics,
        "n_samples": len(gen_texts),
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before.get("syntax_hash") != after.get("syntax_hash"):
        return {
            "verdict": "INCOMPARABLE",
            "reason": "different syntax_hash (data / base / n / wrapper differ)",
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
        "syntax_hash": before["syntax_hash"],
        "metrics_before": b,
        "metrics_after": a,
        "deltas": deltas,
        "improved_metrics": improved,
        "regressed_metrics": regressed,
    }
