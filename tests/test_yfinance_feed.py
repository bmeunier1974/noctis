"""The yfinance live feed: closed-bar emission, throttling, staleness, and driver wiring.

A scripted **fake** downloader stands in for Yahoo Finance — no network — and a fake clock
drives the throttle/staleness logic deterministically. Covers: newly-closed bars emitted once
in timestamp order with the forming tail held back (and released by ``flush``), previously-
provisional bars emitted once a newer one arrives, fetch throttling by ``min_interval``,
degraded-on-error / degraded-on-staleness with recovery, cross-symbol alignment, the real
frame-splitting parser, and end-to-end parity with the batch driver.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from noctis.data.yfinance.feed import (
    YFinanceBarFeed,
    _split_frames,
    build_yfinance_feed,
)
from noctis.engine import SimulatedSleeper
from noctis.live import BarFeed, SessionConfig, run_trading, run_trading_day
from noctis.strategies import Candidate

from ._data_helpers import make_ohlcv

# --- fakes -------------------------------------------------------------------------------


def _norm_frame(prices: list[float], start: pd.Timestamp | None = None) -> pd.DataFrame:
    """A normalized per-symbol frame: UTC minute index + lowercase OHLCV columns."""
    start = start or pd.Timestamp("2026-01-05 14:30", tz="UTC")
    idx = pd.date_range(start, periods=len(prices), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        },
        index=idx,
    )


class _Scripted:
    """Returns one scripted frame-dict per call, repeating the last once exhausted."""

    def __init__(self, per_call: list[dict[str, pd.DataFrame]]):
        self._per_call = list(per_call)
        self.calls = 0

    def __call__(self, symbols, interval):
        self.calls += 1
        return self._per_call[min(self.calls - 1, len(self._per_call) - 1)]


class _Clock:
    """A hand-cranked monotonic clock."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _drain(feed: YFinanceBarFeed, polls: int) -> list[tuple[str, int]]:
    """Poll ``polls`` times; return (symbol, ts_event) for every emitted bar, in order."""
    out: list[tuple[str, int]] = []
    for _ in range(polls):
        for sym, bar in feed.poll_once().items():
            out.append((sym, bar.ts_event))
    return out


# --- 1) closed bars emitted once, in order, forming tail held back then flushed -------------


def test_emits_closed_bars_in_order_dropping_provisional():
    frame = _norm_frame([100.0, 101.0, 102.0])  # t0, t1, t2
    dl = _Scripted([{"AAPL": frame}])
    feed = YFinanceBarFeed(["AAPL"], min_interval=0.0, downloader=dl, clock=_Clock())

    emitted = _drain(feed, polls=4)
    ts = [t for _, t in emitted]

    # Only the two *closed* bars come out of the poll loop, oldest first; t2 is held back.
    assert ts == sorted(ts)
    assert len(ts) == 2
    provisional_ts = int(frame.index[-1].value)
    assert provisional_ts not in ts

    # flush releases the now-complete tail exactly once.
    flushed = feed.flush()
    assert set(flushed) == {"AAPL"}
    assert flushed["AAPL"].ts_event == provisional_ts
    assert feed.flush() == {}  # not emitted twice


def test_ohlcv_values_survive_the_round_trip():
    frame = _norm_frame([100.0, 101.0])
    feed = YFinanceBarFeed(
        ["AAPL"], min_interval=0.0, downloader=_Scripted([{"AAPL": frame}]), clock=_Clock()
    )
    bar = feed.poll_once()["AAPL"]
    row = frame.iloc[0]
    assert bar.ts_event == int(frame.index[0].value)
    assert bar.open == pytest.approx(row["open"])
    assert bar.high == pytest.approx(row["high"])
    assert bar.low == pytest.approx(row["low"])
    assert bar.close == pytest.approx(row["close"])
    assert bar.volume == pytest.approx(row["volume"])


def test_previously_provisional_bar_emits_once_newer_arrives():
    small = _norm_frame([100.0, 101.0])  # t0, t1  (t1 provisional)
    grown = _norm_frame([100.0, 101.0, 102.0, 103.0])  # t0..t3 (t3 provisional)
    dl = _Scripted([{"AAPL": small}, {"AAPL": grown}])
    feed = YFinanceBarFeed(["AAPL"], min_interval=0.0, downloader=dl, clock=_Clock())

    ts = [t for _, t in _drain(feed, polls=6)]
    expected = [int(grown.index[i].value) for i in range(3)]  # t0, t1, t2 — t3 stays provisional
    assert ts == expected


# --- 2) fetches are throttled independently of poll cadence --------------------------------


def test_throttles_fetches_by_min_interval():
    clock = _Clock(1000.0)
    dl = _Scripted([{"AAPL": _norm_frame([100.0, 101.0])}])
    feed = YFinanceBarFeed(["AAPL"], min_interval=30.0, downloader=dl, clock=clock)

    feed.poll_once()  # first poll always fetches
    for _ in range(10):  # many polls inside the throttle window → still one fetch
        feed.poll_once()
    assert dl.calls == 1

    clock.t = 1030.0  # throttle elapsed
    feed.poll_once()
    assert dl.calls == 2


# --- 3) degraded on fetch error and on staleness, each recovering ---------------------------


class _ErrThenOk:
    def __init__(self, ok: dict[str, pd.DataFrame]):
        self.calls = 0
        self._ok = ok

    def __call__(self, symbols, interval):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("yahoo unavailable")
        return self._ok


def test_degraded_on_fetch_error_then_recovers():
    dl = _ErrThenOk({"AAPL": _norm_frame([100.0, 101.0])})
    feed = YFinanceBarFeed(["AAPL"], min_interval=0.0, downloader=dl, clock=_Clock())

    feed.poll_once()
    assert feed.degraded is True  # the failed fetch degraded the feed

    got = feed.poll_once()
    assert feed.degraded is False  # a good fetch with a fresh bar recovers it
    assert set(got) == {"AAPL"}


def test_degraded_on_staleness_then_recovers():
    clock = _Clock(0.0)
    static = {"AAPL": _norm_frame([100.0, 101.0])}
    fresh = {"AAPL": _norm_frame([100.0, 101.0, 102.0, 103.0])}
    # The frame stays static (no new closed bars) across the stale window, then grows.
    dl = _Scripted([static, static, static, fresh])
    feed = YFinanceBarFeed(
        ["AAPL"], min_interval=0.0, stale_after_s=100.0, downloader=dl, clock=clock
    )

    feed.poll_once()  # commits t0 at clock 0
    assert feed.degraded is False

    clock.t = 50.0
    feed.poll_once()  # still the static frame → no new bar, but within the staleness window
    assert feed.degraded is False

    clock.t = 150.0
    feed.poll_once()  # no fresh bar for >100s → stale
    assert feed.degraded is True

    clock.t = 160.0
    feed.poll_once()  # the grown frame delivers a fresh bar → recovered
    assert feed.degraded is False


# --- 4) cross-symbol alignment: one timestamp group per poll -------------------------------


def test_multi_symbol_bars_share_a_group():
    frame = _norm_frame([100.0, 101.0, 102.0])
    dl = _Scripted([{"AAPL": frame, "MSFT": frame}])
    feed = YFinanceBarFeed(["AAPL", "MSFT"], min_interval=0.0, downloader=dl, clock=_Clock())

    group = feed.poll_once()
    assert set(group) == {"AAPL", "MSFT"}  # the same minute for both symbols in one poll
    assert group["AAPL"].ts_event == group["MSFT"].ts_event


def test_feed_satisfies_the_barfeed_protocol():
    feed = YFinanceBarFeed(["MSFT", "AAPL"], downloader=_Scripted([{}]))
    assert isinstance(feed, BarFeed)
    assert set(feed.symbols) == {"AAPL", "MSFT"}
    assert feed.exhausted is False  # clock-bounded: only the session close ends a live day


# --- 5) the real frame-splitting parser (no network) ---------------------------------------


def test_split_frames_handles_grouped_multiindex_and_drops_nan_close():
    idx = pd.date_range("2026-01-05 14:30", periods=3, freq="1min", tz="UTC")
    cols = pd.MultiIndex.from_tuples(
        [(s, f) for s in ("AAPL", "MSFT") for f in ("Open", "High", "Low", "Close", "Volume")]
    )
    raw = pd.DataFrame(0.0, index=idx, columns=cols)
    raw[("AAPL", "Close")] = [100.0, float("nan"), 102.0]  # a no-trade minute for AAPL
    raw[("MSFT", "Close")] = [200.0, 201.0, 202.0]

    frames = _split_frames(raw, ["AAPL", "MSFT"])
    assert set(frames) == {"AAPL", "MSFT"}
    assert list(frames["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(frames["AAPL"]) == 2  # the NaN-close row was dropped
    assert len(frames["MSFT"]) == 3


def test_split_frames_handles_single_symbol_flat_columns():
    idx = pd.date_range("2026-01-05 14:30", periods=2, freq="1min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [10, 20],
        },
        index=idx,
    )
    frames = _split_frames(raw, ["AAPL"])
    assert list(frames["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(frames["AAPL"]) == 2


# --- 6) build factory wires the injected downloader ----------------------------------------


def test_build_yfinance_feed_uses_injected_downloader():
    dl = _Scripted([{"AAPL": _norm_frame([100.0, 101.0])}])
    feed = build_yfinance_feed(symbols=["AAPL"], min_interval=0.0, downloader=dl)
    assert isinstance(feed, YFinanceBarFeed)
    assert set(feed.poll_once()) == {"AAPL"}
    assert dl.calls == 1


# --- 7) end-to-end: the real feed drives the streaming driver like the batch driver ---------


def _norm_from_canonical(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(df["ts_event"], unit="ns", utc=True))
    return pd.DataFrame(
        {
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        },
        index=idx,
    )


def test_streaming_driver_matches_batch_over_yfinance_feed():
    bars = make_ohlcv([100.0 + i * 0.5 for i in range(120)])
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]

    batch = run_trading(candidates=candidates, bars_by_symbol={"AAPL": bars})

    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    # Static frame + zero throttle + frozen clock: the feed refetches cheaply each poll, never
    # goes stale, and (polls drain the closed bars in order; flush releases the final one).
    dl = _Scripted([{"AAPL": _norm_from_canonical(bars)}])
    feed = YFinanceBarFeed(["AAPL"], min_interval=0.0, downloader=dl, clock=_Clock())

    live = run_trading_day(
        SessionConfig(candidates=candidates),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=len(bars) + 20),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
    )

    assert live.summary.bars_processed == len(bars)
    assert live.summary.orders_submitted == batch.orders_submitted
    assert live.summary.fills == batch.fills
    assert live.summary.positions == batch.positions
    assert live.summary.final_equity == pytest.approx(batch.final_equity)
