"""Pure-function regression tests for the Qwen-base adaptation path.

No model download, no GPU, no data files needed.
"""

from __future__ import annotations

from autotrain.qwen_sft.adapt import compare as adapt_compare
from autotrain.qwen_sft.data import _ABS_PATH_RE
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
