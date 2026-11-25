"""Unit tests for the serving benchmark's pure helpers."""

from __future__ import annotations

from benchmarks.serving_bench import percentile


def test_percentile_edges():
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.99) == 10.0


def test_percentile_is_order_insensitive_and_bounded():
    values = [0.4, 0.1, 0.3, 0.2, 0.5]
    assert percentile(values, 0.50) == 0.3
    assert percentile(values, 0.95) == 0.5
    assert percentile(values, 0.99) == 0.5


def test_percentile_never_exceeds_max():
    values = [1.0, 2.0, 3.0, 4.0]
    for q in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert percentile(values, q) <= max(values)
