"""Walk-forward splitter — rolling (train, test) windows, test strictly after train.

Windows are positional (bar-index) ranges so any ordered bar series can be split. Within a
split, train and test never overlap and the test window is always immediately after the
train window; consecutive splits advance by ``step``. A calendar helper derives bar counts
from month/day lengths when needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Half-open positional ranges: train ``[train_start, train_end)``, test after it."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def walk_forward(n: int, train_size: int, test_size: int, step: int) -> list[Split]:
    """Rolling walk-forward splits over ``n`` ordered bars.

    Each split has a ``train_size``-bar training window immediately followed by a
    ``test_size``-bar test window; the start advances by ``step`` bars per split. Only
    splits that fully fit within ``n`` bars are returned.
    """
    if min(train_size, test_size, step) <= 0:
        raise ValueError("train_size, test_size, and step must be positive")
    splits: list[Split] = []
    start = 0
    idx = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        splits.append(Split(idx, start, train_end, train_end, test_end))
        start += step
        idx += 1
    return splits
