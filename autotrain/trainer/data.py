"""Token data loading with resumable cursor.

Reads flat binary token files (uint16), yields fixed-length batches.
The cursor (absolute token offset) is part of the training state so a
resumed run continues exactly where it stopped, never re-reading and
never skipping data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Flat uint16 token stream with train/val split by fraction of file.

    An optional uint8 ``label_file`` marks which tokens are answer tokens
    (1) vs question/padding (0). When present, target positions whose label
    is 0 are mapped to ``-1`` so ``cross_entropy(ignore_index=-1)`` skips
    them -- this is answer-side loss, not diluted next-token prediction.
    """

    def __init__(
        self,
        token_file: str | Path,
        val_fraction: float = 0.01,
        label_file: str | Path | None = None,
    ):
        self.path = Path(token_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"token file not found: {self.path}")
        n_tokens = self.path.stat().st_size // 2  # uint16
        if n_tokens < 1024:
            raise ValueError(f"token file too small: {n_tokens} tokens")
        self.n_total = n_tokens
        self.n_val = max(1, int(n_tokens * val_fraction))
        self.n_train = n_tokens - self.n_val
        self.label_path = Path(label_file) if label_file is not None else None
        if self.label_path is not None:
            if not self.label_path.is_file():
                raise FileNotFoundError(f"label file not found: {self.label_path}")
            n_labels = self.label_path.stat().st_size  # uint8
            if n_labels != n_tokens:
                raise ValueError(
                    f"label/token length mismatch: {n_labels} labels vs {n_tokens} tokens"
                )

    def _read(self, start: int, count: int) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(start * 2)
            raw = f.read(count * 2)
        if len(raw) < count * 2:
            raise EOFError("unexpected end of token file")
        return np.frombuffer(raw, dtype=np.uint16)

    def _mask_targets(self, y_buf: np.ndarray, label_start: int) -> np.ndarray:
        """Return copied targets with non-answer positions set to -1.

        ``y_buf[t]`` is the target for position ``t`` in the batch window,
        corresponding to source token ``label_start + t + 1`` (shift by one).
        Labels are read at that shifted index.
        """
        if self.label_path is None:
            return y_buf.astype(np.int64)
        with open(self.label_path, "rb") as f:
            f.seek(label_start + 1)  # +1 for the next-token shift
            raw = f.read(y_buf.size)
        if len(raw) < y_buf.size:
            raise EOFError("unexpected end of label file")
        labels = np.frombuffer(raw, dtype=np.uint8)
        y = y_buf.astype(np.int64)
        y[labels == 0] = -1
        return y

    def get_train_batch(self, cursor: int, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (inputs, targets, next_cursor).

        Inputs are tokens [cursor, cursor + batch*seq_len), targets are the
        same shifted by one. Wraps around to 0 when reaching the end of the
        training region.
        """
        need = batch_size * seq_len + 1
        if cursor + need > self.n_train:
            cursor = 0
        buf = self._read(cursor, need).astype(np.int64)
        x = torch.from_numpy(buf[:-1].reshape(batch_size, seq_len))
        y = self._mask_targets(buf[1:], label_start=cursor)
        y = torch.from_numpy(y.reshape(batch_size, seq_len))
        next_cursor = cursor + batch_size * seq_len
        return x, y, next_cursor

    def val_window_count(self, need: int, requested: int) -> int:
        """How many disjoint ``need``-token windows fit in the val region.

        Returns at most ``requested``, but never more than physically fit
        without overlapping. This is the guard that keeps a validation pass
        representative: the old code sampled 4 windows whose start offsets
        differed by only 3 tokens (~4.1K independent tokens total), so
        EarlyStopTracker was effectively re-reading the same tiny slice over
        and over. On the real ~312K-token tail this returns the full
        ``requested`` count; on small test fixtures it shrinks gracefully.
        """
        usable = max(1, self.n_val - need)
        return max(1, min(max(1, requested), usable // need + 1))

    def get_val_batch(
        self, batch_size: int, seq_len: int, offset: int = 0,
        n_windows: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic validation batch from the tail region.

        ``offset`` is a **window index** when ``n_windows > 1``: the val
        region is split into evenly-spaced, non-overlapping windows (capped
        by :meth:`val_window_count`), so one validation pass samples a wide,
        representative slice of the tail instead of re-reading ~4K tokens.
        This matters because EarlyStopTracker compares val loss across steps
        -- a tiny, near-duplicate sample makes that comparison noise-dominated.

        With the default ``n_windows=1`` the legacy behaviour is preserved
        (``offset`` is a raw token offset), keeping callers and tests that
        pass a single offset unchanged.
        """
        need = batch_size * seq_len + 1
        usable = max(1, self.n_val - need)
        if n_windows > 1:
            k = self.val_window_count(need, n_windows)
            stride = usable // (k - 1) if k > 1 else 0
            pos = min((offset % k) * stride, usable)
            start = self.n_train + pos
        else:
            start = self.n_train + (offset % usable)
        if start + need > self.n_total:
            start = self.n_total - need
        buf = self._read(start, need).astype(np.int64)
        x = torch.from_numpy(buf[:-1].reshape(batch_size, seq_len))
        y = self._mask_targets(buf[1:], label_start=start)
        y = torch.from_numpy(y.reshape(batch_size, seq_len))
        return x, y

    @staticmethod
    def write_token_file(path: str | Path, tokens: np.ndarray) -> None:
        """Helper for tests and data pipeline: write uint16 token file."""
        arr = np.asarray(tokens, dtype=np.uint16)
        with open(path, "wb") as f:
            f.write(arr.tobytes())

    @staticmethod
    def token_count(path: str | Path) -> int:
        return Path(path).stat().st_size // struct.calcsize("H")
