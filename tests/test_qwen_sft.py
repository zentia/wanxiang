"""Pure-function regression tests for the Qwen-base adaptation path.

No model download, no GPU, no data files needed.
"""

from __future__ import annotations

from autotrain.qwen_sft.adapt import compare as adapt_compare
from autotrain.qwen_sft.data import _ABS_PATH_RE
from autotrain.qwen_sft.quality import (
    build_spec as quality_build_spec,
    code_char_ratio,
    compare as quality_compare,
    explanation_leak,
    pair_metrics,
    raw_char_f1,
)
from autotrain.qwen_sft.ruler import build_spec


def _report(adapt_hash: str, ce: float, top1: float) -> dict:
    return {
        "adapt_hash": adapt_hash,
        "answer_cross_entropy": ce,
        "answer_top1_accuracy": top1,
    }


def test_adapt_compare_rejects_different_rulers():
    result = adapt_compare(_report("aaa", 1.0, 0.5), _report("bbb", 1.0, 0.5))
    assert result["verdict"] == "INCOMPARABLE"


def test_adapt_compare_verdicts():
    base = _report("same", 1.77, 0.684)
    better = _report("same", 1.32, 0.730)
    worse = _report("same", 2.10, 0.600)
    assert adapt_compare(base, better)["verdict"] == "IMPROVED"
    assert adapt_compare(base, worse)["verdict"] == "REGRESSED"
    assert adapt_compare(base, base)["verdict"] == "NO_CHANGE"


def test_ruler_spec_is_deterministic_and_base_sensitive():
    a = build_spec("prompt_sha", "1.5b", n=16, gen_len=64)
    b = build_spec("prompt_sha", "1.5b", n=16, gen_len=64)
    c = build_spec("prompt_sha", "7b", n=16, gen_len=64)
    assert a == b
    assert a["gen_hash"] != c["gen_hash"], "different bases must be incomparable"


def test_abs_path_regex_catches_private_paths():
    for blob in (
        r'{"root": "X:\\private\\src"}',
        r'{"root": "/Users/someone/project"}',
        r'{"root": "/home/someone/project"}',
        r'{"root": "D:/private/src"}',
    ):
        assert _ABS_PATH_RE.search(blob), blob
    assert not _ABS_PATH_RE.search('{"note": "relative runs/data only"}')


# ---------------------------------------------------------------------------
# Format / correctness-proxy ruler
# ---------------------------------------------------------------------------


def test_explanation_leak_detects_chatty_replies_only():
    assert explanation_leak(
        "Certainly! It looks like you're working on a C# method."
    ) == 1.0
    assert explanation_leak(
        "        return;\n\n    m_Texture.Apply();\n}\n"
    ) == 0.0


def test_code_char_ratio_and_indent_favor_code():
    code = "        return;\n    m_Texture.Apply();\n}\n"
    prose = "This method updates the texture when it is available"
    assert code_char_ratio(code) > code_char_ratio(prose)
    assert pair_metrics(code, code)["indented_line_ratio"] > 0.0


def test_raw_char_f1_keeps_punctuation():
    assert raw_char_f1("a;\n", "a;\n") == 1.0
    assert raw_char_f1("abc", "xyz") == 0.0
    # Unlike eval2's normalizing char_f1, punctuation/whitespace count here.
    assert raw_char_f1("a;", "a ") < 1.0


def test_pair_metrics_has_all_keys_and_is_pure():
    m = pair_metrics("        return;\n}\n", "        return;\n}\n")
    for k in ("explanation_leak_rate", "code_char_ratio", "code_char_ratio_gap",
              "indented_line_ratio", "raw_char_f1", "char_f1", "rouge_l",
              "edit_similarity", "exact_match", "composite"):
        assert k in m
    assert m["exact_match"] == 1.0
    assert m["code_char_ratio_gap"] == 0.0


def _qreport(qhash: str, **metrics) -> dict:
    base = {"quality_hash": qhash, "metrics": {
        "explanation_leak_rate": 0.0, "code_char_ratio": 0.3,
        "code_char_ratio_gap": 0.1, "indented_line_ratio": 0.5,
        "raw_char_f1": 0.4, "char_f1": 0.4, "rouge_l": 0.4,
        "edit_similarity": 0.4, "exact_match": 0.0, "composite": 0.4,
    }}
    base["metrics"].update(metrics)
    return base


def test_quality_compare_rejects_different_rulers():
    assert quality_compare(_qreport("a"), _qreport("b"))["verdict"] == "INCOMPARABLE"


def test_quality_compare_verdicts():
    base = _qreport("same")
    better = _qreport("same", explanation_leak_rate=0.0, char_f1=0.8,
                      code_char_ratio_gap=0.0, raw_char_f1=0.8,
                      composite=0.8, exact_match=0.1)
    worse = _qreport("same", explanation_leak_rate=1.0, char_f1=0.1,
                     code_char_ratio_gap=0.5, raw_char_f1=0.1)
    assert quality_compare(base, better)["verdict"] == "IMPROVED"
    assert quality_compare(base, worse)["verdict"] == "REGRESSED"
    assert quality_compare(base, _qreport("same"))["verdict"] == "NO_CHANGE"


def test_quality_spec_is_deterministic_and_base_sensitive():
    a = quality_build_spec("sha", "corpus", "7b", n=32, gen_len=64)
    b = quality_build_spec("sha", "corpus", "7b", n=32, gen_len=64)
    c = quality_build_spec("sha", "corpus", "1.5b", n=32, gen_len=64)
    assert a == b
    assert a["quality_hash"] != c["quality_hash"]
