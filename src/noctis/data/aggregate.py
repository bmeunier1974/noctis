"""Bar aggregation: the lake stores one canonical granularity, strategies pick their own.

Ingestion always fetches 1-minute bars (one vendor schema, one storage format); a strategy
declares the ``timeframe`` its thesis needs and the engine resamples on read. Two code
paths, one bucketing rule, so research and live can never disagree:

* :func:`aggregate_bars` — vectorised, whole-frame resample for the backtest pipeline.
* :class:`StreamingAggregator` — incremental proxy for the live loop: buffers fine bars
  and emits one aggregated :class:`~noctis.strategies.base.Bar` when its bucket completes
  (i.e. when the first bar of the NEXT bucket arrives — robust to missing minutes).

An aggregated bar carries the ``ts_event`` of its LAST constituent bar: the bar "exists"
only once fully observed, so replaying aggregated frames can never leak the future. The
native timeframe ("1m") is a pure pass-through on both paths.
"""

from __future__ import annotations

import pandas as pd

from noctis.strategies.base import Bar

_MINUTE_NS = 60 * 1_000_000_000

# Supported strategy timeframes → bucket width in UTC nanoseconds. "1d" buckets by UTC
# calendar day, which equals the trading session for US equities (RTH never crosses UTC
# midnight).
TIMEFRAMES: dict[str, int] = {
    "1m": _MINUTE_NS,
    "5m": 5 * _MINUTE_NS,
    "15m": 15 * _MINUTE_NS,
    "30m": 30 * _MINUTE_NS,
    "1h": 60 * _MINUTE_NS,
    "1d": 24 * 60 * _MINUTE_NS,
}

NATIVE_TIMEFRAME = "1m"

# Bars per year per timeframe for metric annualization (252 sessions × 390 RTH minutes).
BARS_PER_YEAR: dict[str, int] = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h": round(252 * 6.5),
    "1d": 252,
}


def validate_timeframe(timeframe: str) -> str:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}; supported: {sorted(TIMEFRAMES)}")
    return timeframe


def bars_per_year(timeframe: str) -> int:
    return BARS_PER_YEAR[validate_timeframe(timeframe)]


def aggregate_bars(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a 1-minute OHLCV frame to ``timeframe`` (pass-through copy for "1m").

    Buckets are fixed UTC windows (``ts_event // bucket_ns``); each output row is
    open=first / high=max / low=min / close=last / volume=sum of its constituents, stamped
    with the last constituent's ``ts_event``. A trailing partial bucket is included — the
    walk-forward splitter slices whatever it is given, and live parity is preserved
    because only the final bar can differ.
    """
    validate_timeframe(timeframe)
    if timeframe == NATIVE_TIMEFRAME or len(bars) == 0:
        return bars.reset_index(drop=True).copy()
    bucket = bars["ts_event"].astype("int64") // TIMEFRAMES[timeframe]
    grouped = bars.groupby(bucket.values, sort=True)
    out = grouped.agg(
        ts_event=("ts_event", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return out.reset_index(drop=True)


def last_completed_htf(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Per base row, the OHLCV of the last *fully completed* ``timeframe`` bucket.

    The vectorised twin of :class:`~noctis.strategies.indicators.HtfBars` for ``signals()``
    overrides: each base row sees the higher-timeframe bar whose bucket has already closed,
    matched on ``bucket(base_ts) > bucket(htf_row)`` — strictly greater-than, so the
    *in-progress* bucket (which :func:`aggregate_bars` includes as a trailing partial) is
    never visible. Base rows before the first bucket closes are ``NaN``.

    Returns a frame indexed like ``bars`` with columns ``ts_event, open, high, low, close,
    volume`` describing that completed bucket (``ts_event`` is the completed bucket's last
    constituent stamp). Same completed-bucket view the streaming path sees bar-by-bar, so the
    two paths agree on both the values and *when* each becomes visible.
    """
    validate_timeframe(timeframe)
    cols = ["ts_event", "open", "high", "low", "close", "volume"]
    n = len(bars)
    out = pd.DataFrame({c: pd.Series([float("nan")] * n, dtype="float64") for c in cols})
    if n == 0:
        return out
    completed = aggregate_bars(bars, timeframe)  # every bucket, incl. the trailing partial
    bucket_ns = TIMEFRAMES[timeframe]
    base_bucket = (bars["ts_event"].astype("int64") // bucket_ns).to_numpy()
    htf_bucket = (completed["ts_event"].astype("int64") // bucket_ns).to_numpy()
    # For each base row, the last completed bucket is the newest HTF row whose bucket index
    # is strictly less than the base row's bucket index.
    pos = htf_bucket.searchsorted(base_bucket, side="left") - 1
    visible = pos >= 0
    src = pos[visible]
    for c in cols:
        col_vals = completed[c].to_numpy()
        out.loc[visible, c] = col_vals[src]
    return out


class StreamingAggregator:
    """Incremental 1m→timeframe proxy for the live loop.

    ``add(bar)`` buffers the fine bar and returns the previous bucket's aggregated
    :class:`Bar` when ``bar`` opens a new bucket, else ``None``. The native timeframe
    returns every bar immediately (no delay), so existing 1m behavior is untouched. A
    session-final partial bucket is deliberately never emitted: the backtest never scored
    partial buckets, so live must not act on them either.
    """

    def __init__(self, timeframe: str):
        self.timeframe = validate_timeframe(timeframe)
        self._bucket_ns = TIMEFRAMES[timeframe]
        self._bucket: int | None = None
        self._agg: Bar | None = None  # running aggregate of the open bucket

    def add(self, bar: Bar) -> Bar | None:
        if self.timeframe == NATIVE_TIMEFRAME:
            return bar
        bucket = bar.ts_event // self._bucket_ns
        emitted: Bar | None = None
        if self._agg is not None and bucket != self._bucket:
            emitted = self._agg
            self._agg = None
        if self._agg is None:
            self._bucket = bucket
            self._agg = Bar(bar.ts_event, bar.open, bar.high, bar.low, bar.close, bar.volume)
        else:
            prev = self._agg
            self._agg = Bar(
                ts_event=bar.ts_event,
                open=prev.open,
                high=max(prev.high, bar.high),
                low=min(prev.low, bar.low),
                close=bar.close,
                volume=prev.volume + bar.volume,
            )
        return emitted
