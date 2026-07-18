"""A call-counting mock vendor + deterministic bar generators for the tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd

from noctis.data.types import (
    BAR_COLUMNS,
    NS_PER_SECOND,
    day_start_ns,
    empty_bars,
    ns_to_date,
)


def make_ohlcv(closes: Sequence[float], spread: float = 0.5) -> pd.DataFrame:
    """Build an OHLCV frame from a close series (high/low bracket close by ``spread``)."""
    close = np.asarray(closes, dtype="float64")
    n = len(close)
    ts = [i * 60 * NS_PER_SECOND for i in range(n)]
    return pd.DataFrame(
        {
            "ts_event": ts,
            "open": close.copy(),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": [1000] * n,
        }
    )


def price_series(n: int = 200, seed: int = 7) -> np.ndarray:
    """A deterministic oscillating price path that exercises entries and exits."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, n).cumsum()
    trend = np.linspace(0.0, 10.0, n)
    wave = 5.0 * np.sin(np.linspace(0.0, 6.0 * np.pi, n))
    return 100.0 + trend + wave + steps


def bars_for_range(symbol: str, start: int, end: int) -> pd.DataFrame:
    """One deterministic bar per weekday at 14:00 UTC within ``[start, end]`` (inclusive)."""
    rows = []
    day = ns_to_date(start)
    last_day = ns_to_date(end)
    while day <= last_day:
        if day.weekday() < 5:  # Mon–Fri
            ts = day_start_ns(day) + 14 * 3600 * NS_PER_SECOND
            if start <= ts <= end:
                base = 100.0 + (abs(hash((symbol, day.toordinal()))) % 10)
                rows.append((ts, base, base + 1.0, base - 1.0, base + 0.5, 1000))
        day += timedelta(days=1)
    if not rows:
        return empty_bars()
    return pd.DataFrame(rows, columns=list(BAR_COLUMNS))


class MockVendor:
    """Counts cost + fetch calls and records fetched ranges so tests can assert them."""

    def __init__(self, cost_per_day: float = 0.001):
        self.cost_per_day = cost_per_day
        self.cost_calls = 0
        self.fetch_calls = 0
        self.fetch_ranges: list[tuple[int, int]] = []

    def _days(self, start: int, end: int) -> int:
        return max(1, (ns_to_date(end) - ns_to_date(start)).days + 1)

    def get_cost(self, *, dataset, schema, symbol, start, end) -> float:
        self.cost_calls += 1
        return self.cost_per_day * self._days(start, end)

    def fetch_bars(self, *, dataset, schema, symbol, start, end) -> pd.DataFrame:
        self.fetch_calls += 1
        self.fetch_ranges.append((start, end))
        return bars_for_range(symbol, start, end)
