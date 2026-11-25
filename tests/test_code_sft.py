"""Pure-function regression tests for the restructured code SFT data and the
objective generation ruler. No model, no GPU, no source files needed.
"""

from __future__ import annotations

from autotrain.code_corpus.sft import (
    _find_blocks,
    _is_template,
    _structural_samples,
    _window_samples,
)
from autotrain.eval_gen.runner import compare, sequence_metrics


# ---------------------------------------------------------------------------
# Block-aligned sample extraction
# ---------------------------------------------------------------------------


def test_find_blocks_aligns_to_braces_and_never_slices_mid_block():
    lines = [
        "public void A() {",       # 0
        "    int x = 1;",          # 1
        "    if (x > 0) {",        # 2
        "        x++;",            # 3
        "    }",                   # 4
        "    return;",             # 5
        "}",                       # 6
        "public void B() {",       # 7
        "    x = 2;",              # 8
        "    x += 3;",             # 9
        "    Log(x);",             # 10
        "}",                       # 11
    ]
    blocks = list(_find_blocks(lines))
    assert (0, 7) in blocks
    assert (7, 12) in blocks
    for start, end in blocks:
        assert lines[start].count("{") > lines[start].count("}")
        assert sum(l.count("{") - l.count("}") for l in lines[start:end]) == 0


def test_is_template_rejects_boilerplate():
    assert _is_template(["// a", "// b", "// c", "// d", "// e"]) is True
    assert _is_template(["public int X { get; set; }", "a", "b", "c", "d"]) is True
    assert _is_template(["same"] * 6) is True
    good = ["void F() {", "  int x = 1;", "  x += 2;", "  Log(x);", "}"]
    assert _is_template(good) is False


def test_structural_samples_deduplicate_answers():
    body = ["void A() {", "  int x = 1;", "  x += 2;", "  Log(x);", "  return;", "}"]
    lines = body + ["void B() {"] + body[1:]
    seen: set[str] = set()
    answers = [a for _p, a in _structural_samples(lines, seen=seen)]
    assert answers, "should produce at least one sample"
    assert len(answers) == len(set(answers)), "duplicate answers must be dropped"


def test_structural_samples_have_variable_prompt_and_shorter_answers():
    lines = [
        "void F() {", "  int a = 1;", "  int b = 2;", "  int c = 3;",
        "  a += b;", "  b += c;", "  Log(a);", "  Log(b);", "  return;", "}",
    ]
    samples = list(_structural_samples(lines))
    assert samples
    for prefix, answer in samples:
        assert prefix.strip() and answer.strip()
        assert len(answer.split("\n")) <= 10


def test_window_mode_still_reproduces_legacy_fixed_lines():
    lines = [f"line {i}" for i in range(40)]
    samples = list(_window_samples(lines, prefix_lines=8, answer_lines=12, stride=32))
    assert len(samples) == 1
    assert len(samples[0][0].split("\n")) == 8
    assert len(samples[0][1].split("\n")) == 12


# ---------------------------------------------------------------------------
# Objective generation ruler
# ---------------------------------------------------------------------------


def test_sequence_metrics_detects_a_loop():
    loop = [1, 2, 3] * 8
    m = sequence_metrics(loop)
    assert m["looped"] == 1.0
    assert m["repeat_3gram_rate"] > 0.5
    assert m["distinct_1"] < 0.2


def test_sequence_metrics_on_unique_tokens():
    m = sequence_metrics(list(range(64)))
    assert m["looped"] == 0.0
    assert m["repeat_3gram_rate"] == 0.0
    assert m["distinct_1"] == 1.0
    assert m["max_repeat_run"] == 1.0


def test_compare_rejects_different_rulers():
    empty = {k: 0.0 for k in ("distinct_1", "distinct_2", "repeat_3gram_rate",
                              "max_3gram_count", "max_repeat_run", "looped")}
    result = compare(
        {"gen_hash": "aaa", "metrics": empty},
        {"gen_hash": "bbb", "metrics": empty},
    )
    assert result["verdict"] == "INCOMPARABLE"
