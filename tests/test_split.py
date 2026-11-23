"""Word-level holdout split + shared answer cleaning."""

from __future__ import annotations

from autotrain.split import (MAX_ANSWER_CHARS, answer_for_row, clean_answer,
                             holdout_words, split_rows)

ROWS = [
    {"word": f"词{i}", "explanation": f"第{i}个意思。"}
    for i in range(400)
]


class TestHoldoutWords:
    def test_deterministic(self):
        assert holdout_words(ROWS, seed=1, count=50) == holdout_words(ROWS, seed=1, count=50)

    def test_seed_changes_split(self):
        assert holdout_words(ROWS, seed=1, count=50) != holdout_words(ROWS, seed=2, count=50)

    def test_count_and_unique(self):
        w = holdout_words(ROWS, seed=7, count=30)
        assert len(w) == 30 and len(set(w)) == 30

    def test_split_disjoint_and_complete(self):
        held = set(holdout_words(ROWS, seed=3, count=20))
        train, hold = split_rows(ROWS, held)
        assert len(train) + len(hold) == len(ROWS)
        assert {r["word"] for r in hold} == held
        assert not ({r["word"] for r in train} & held)

    def test_too_few_words_raises(self):
        import pytest
        with pytest.raises(ValueError):
            holdout_words(ROWS[:5], seed=1, count=50)


class TestCleanAnswer:
    def test_strips_example_marker(self):
        assert clean_answer("原指人手大◇多用来形容花钱不节省。") == "原指人手大"

    def test_prefers_definition_cue(self):
        out = clean_answer("日中太阳当头。指古代集市交易。")
        assert "指" in out

    def test_drops_bare_mutual_reference(self):
        # 同X on its own is not a definition -- but still returns something usable
        out = clean_answer("同画蛇添足。")
        assert out is not None
        assert "画蛇添足" in out

    def test_strips_quote_marks(self):
        out = clean_answer("情感表现在言辞当中。同情见乎辞”。")
        assert "”" not in out

    def test_truncates(self):
        out = clean_answer("指" + "很长" * 40)
        assert len(out) <= MAX_ANSWER_CHARS

    def test_empty_returns_none(self):
        assert clean_answer("") is None
        assert clean_answer("◇。；") is None

    def test_answer_for_row_priority(self):
        row = {"explanation": "解释释义", "derivation": "典故出处", "example": "例句说明"}
        assert answer_for_row(row) == "解释释义"
        assert answer_for_row({"explanation": "", "derivation": "典出此处"}) == "典出此处"
