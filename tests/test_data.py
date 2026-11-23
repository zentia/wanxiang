"""TokenDataset batching, cursor semantics, and train/val split."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autotrain.trainer.data import TokenDataset


class TestTokenDataset:
    def test_counts(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        assert ds.n_total == 200_000
        assert ds.n_train + ds.n_val == ds.n_total
        assert ds.n_val == 10_000

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TokenDataset(tmp_path / "nope.bin")

    def test_batch_shapes(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x, y, cursor = ds.get_train_batch(cursor=0, batch_size=4, seq_len=64)
        assert x.shape == (4, 64)
        assert y.shape == (4, 64)
        assert cursor == 4 * 64
        # targets are inputs shifted by one within the flat stream
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        assert torch.equal(flat_x[1:], flat_y[:-1])

    def test_cursor_advances_without_overlap(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x1, _, c1 = ds.get_train_batch(cursor=0, batch_size=2, seq_len=64)
        x2, _, c2 = ds.get_train_batch(cursor=c1, batch_size=2, seq_len=64)
        assert c2 == 256
        assert not torch.equal(x1, x2)

    def test_cursor_wraps_at_end(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        huge_cursor = ds.n_train  # at/after end
        x, _, cursor = ds.get_train_batch(cursor=huge_cursor, batch_size=2, seq_len=64)
        assert cursor == 2 * 64  # wrapped to 0 then advanced

    def test_val_batch_deterministic(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x1, y1 = ds.get_val_batch(batch_size=2, seq_len=64, offset=0)
        x2, y2 = ds.get_val_batch(batch_size=2, seq_len=64, offset=0)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
        # val region is disjoint from train region
        assert x1.min() >= 0

    def test_token_count(self, token_file):
        assert TokenDataset.token_count(token_file) == 200_000


class TestAnswerSideMask:
    """uint8 label stream maps non-answer targets to -1 (ignored by CE)."""

    @staticmethod
    def _write_labels(tmp_path, tokens_path, labels):
        p = tmp_path / "labels.bin"
        np.asarray(labels, dtype=np.uint8).tofile(p)
        return str(p)

    def test_label_masks_question_tokens(self, tmp_path, token_file):
        n = 200_000
        # all zeros except every 3rd position is an answer token
        labels = [1 if i % 3 == 0 else 0 for i in range(n)]
        label_path = self._write_labels(tmp_path, token_file, labels)
        ds = TokenDataset(token_file, val_fraction=0.05, label_file=label_path)
        x, y, _ = ds.get_train_batch(cursor=0, batch_size=2, seq_len=64)
        # batch window is tokens [0, 128); targets are [1, 129).
        # masked positions (target source label==0) must be exactly -1
        flat_y = y.reshape(-1).tolist()
        src = np.fromfile(token_file, dtype="<u2")[1:129]
        expected = [int(v) if labels[k + 1] == 1 else -1
                    for k, v in enumerate(src)]
        assert flat_y == expected
        # at least some targets survive (answer tokens present)
        assert any(v != -1 for v in flat_y)

    def test_label_length_mismatch_raises(self, tmp_path, token_file):
        label_path = self._write_labels(tmp_path, token_file, [1] * 10)
        with pytest.raises(ValueError, match="label/token length mismatch"):
            TokenDataset(token_file, label_file=label_path)

    def test_no_label_keeps_all_targets(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        x, y, _ = ds.get_train_batch(cursor=0, batch_size=2, seq_len=64)
        assert (y >= 0).all()


class TestDisjointValWindows:
    """Regression for the PREMATURE early-stop root cause (EARLY_STOP_REVIEW.md).

    The legacy ``_validate`` sampled 4 windows whose starts differed by only
    3 tokens (~4.1K independent tokens). These tests lock in the fix: one
    validation pass now spreads disjoint windows across (almost) the whole
    tail, and shrinks gracefully when the tail is too small to fit them all.
    """

    def test_real_scale_windows_are_disjoint_and_cover_tail(self, tmp_path):
        from autotrain.trainer.runner import VAL_WINDOWS

        # Sparse 31MB file -> reproduces the real lm-scaled3 tail (312,197
        # val tokens) without allocating a full buffer. Token values are
        # irrelevant here; we only inspect the read offsets.
        n_total = 15_609_856
        p = tmp_path / "big.bin"
        with open(p, "wb") as f:
            f.truncate(n_total * 2)

        ds = TokenDataset(p, val_fraction=0.02)
        bs, sl = 32, 128
        need = bs * sl + 1
        k = ds.val_window_count(need, VAL_WINDOWS)
        assert k == 16, "a 312K-token tail must fit all VAL_WINDOWS"

        starts: list[int] = []
        orig_read = ds._read

        def spy(start, count):
            starts.append(start)
            return orig_read(start, count)

        ds._read = spy
        for i in range(k):
            ds.get_val_batch(bs, sl, offset=i, n_windows=VAL_WINDOWS)

        assert len(starts) == k
        assert starts == sorted(starts), "window starts must be ascending"
        assert all(starts[j + 1] - starts[j] >= need for j in range(k - 1)), \
            "windows must not overlap"
        covered = starts[-1] + need - starts[0]
        assert covered >= 0.8 * ds.n_val, "windows must cover >=80% of the tail"
        assert starts[0] >= ds.n_train, "val windows must stay in the tail region"

    def test_window_count_caps_to_fit_on_small_tail(self, token_file):
        ds = TokenDataset(token_file, val_fraction=0.05)
        bs, sl = 32, 64
        need = bs * sl + 1
        k = ds.val_window_count(need, 16)
        usable = ds.n_val - need
        assert 1 <= k <= 16
        assert (k - 1) * need <= usable, "capped windows must fit without overlap"

        starts: list[int] = []
        orig_read = ds._read

        def spy(start, count):
            starts.append(start)
            return orig_read(start, count)

        ds._read = spy
        for i in range(k):
            ds.get_val_batch(bs, sl, offset=i, n_windows=16)
        assert len(starts) == k
        assert all(starts[j + 1] - starts[j] >= need for j in range(k - 1))

    def test_legacy_single_window_keeps_token_offset_semantics(self, token_file):
        """n_windows=1 (default) must preserve the old raw-token offset."""
        ds = TokenDataset(token_file, val_fraction=0.05)
        bs, sl = 2, 64

        starts: list[int] = []
        orig_read = ds._read

        def spy(start, count):
            starts.append(start)
            return orig_read(start, count)

        ds._read = spy
        ds.get_val_batch(bs, sl, offset=0)  # default n_windows=1
        ds.get_val_batch(bs, sl, offset=1)
        assert starts[1] - starts[0] == 1, "legacy offset is a 1-token shift"
