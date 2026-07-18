"""Bar aggregation: batch resample, the streaming proxy, and their parity.

The lake stores 1m bars; strategies declare a ``timeframe`` and both research (batch
:func:`aggregate_bars`) and live (:class:`StreamingAggregator`) must produce the same
aggregated bars from the same minute stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noctis.data.aggregate import (
    NATIVE_TIMEFRAME,
    StreamingAggregator,
    aggregate_bars,
    bars_per_year,
    last_completed_htf,
)
from noctis.strategies.base import Bar

_NS_PER_MINUTE = 60 * 1_000_000_000


def minute_frame(n: int, seed: int = 0, start_ts: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    return pd.DataFrame(
        {
            "ts_event": [start_ts + i * _NS_PER_MINUTE for i in range(n)],
            "open": close + rng.normal(0.0, 0.1, n),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )


def as_bars(df: pd.DataFrame) -> list[Bar]:
    return [
        Bar(
            int(r.ts_event),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.volume),
        )
        for r in df.itertuples(index=False)
    ]


def test_aggregate_5m_ohlcv_semantics():
    df = minute_frame(10)
    out = aggregate_bars(df, "5m")
    assert len(out) == 2
    first = df.iloc[:5]
    assert out.loc[0, "open"] == first["open"].iloc[0]
    assert out.loc[0, "high"] == first["high"].max()
    assert out.loc[0, "low"] == first["low"].min()
    assert out.loc[0, "close"] == first["close"].iloc[-1]
    assert out.loc[0, "volume"] == first["volume"].sum()
    # Stamped with the LAST constituent's ts: the bar exists only once fully observed.
    assert out.loc[0, "ts_event"] == first["ts_event"].iloc[-1]


def test_aggregate_native_timeframe_is_passthrough_copy():
    df = minute_frame(7)
    out = aggregate_bars(df, NATIVE_TIMEFRAME)
    pd.testing.assert_frame_equal(out, df.reset_index(drop=True))
    assert out is not df


def test_aggregate_trailing_partial_bucket_included():
    assert len(aggregate_bars(minute_frame(12), "5m")) == 3  # 5 + 5 + 2


def test_aggregate_buckets_by_time_not_row_count():
    # A halt (missing minutes) must not shift bucket boundaries.
    df = minute_frame(10).drop(index=[2, 3]).reset_index(drop=True)
    out = aggregate_bars(df, "5m")
    assert len(out) == 2
    assert out.loc[0, "volume"] == df.iloc[:3]["volume"].sum()  # minutes 0,1,4


def test_aggregate_daily_buckets_by_utc_day():
    df = minute_frame(3 * 24 * 60)  # three full UTC days of minutes
    out = aggregate_bars(df, "1d")
    assert len(out) == 3
    assert bars_per_year("1d") == 252


def test_unknown_timeframe_rejected():
    with pytest.raises(ValueError, match="unsupported timeframe"):
        aggregate_bars(minute_frame(5), "7m")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        StreamingAggregator("2h")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        bars_per_year("weekly")


def test_streaming_native_timeframe_emits_immediately():
    agg = StreamingAggregator(NATIVE_TIMEFRAME)
    bar = Bar(0, 1.0, 2.0, 0.5, 1.5, 10.0)
    assert agg.add(bar) is bar  # zero delay: existing 1m behavior untouched


def test_streaming_matches_batch_on_completed_buckets():
    df = minute_frame(37, seed=3)
    batch = aggregate_bars(df, "15m")
    agg = StreamingAggregator("15m")
    emitted: list[Bar] = []
    for bar in as_bars(df):
        done = agg.add(bar)
        if done is not None:
            emitted.append(done)
    # Streaming deliberately never emits the trailing partial bucket; batch includes it.
    assert len(emitted) == len(batch) - 1
    for got, want in zip(emitted, batch.iloc[:-1].itertuples(index=False), strict=True):
        assert got.ts_event == int(want.ts_event)
        assert got.open == pytest.approx(float(want.open))
        assert got.high == pytest.approx(float(want.high))
        assert got.low == pytest.approx(float(want.low))
        assert got.close == pytest.approx(float(want.close))
        assert got.volume == pytest.approx(float(want.volume))


def test_streaming_emits_on_first_bar_of_next_bucket():
    agg = StreamingAggregator("5m")
    bars = as_bars(minute_frame(6))
    assert all(agg.add(b) is None for b in bars[:5])  # bucket still open
    done = agg.add(bars[5])  # minute 5 opens the next bucket
    assert done is not None
    assert done.ts_event == bars[4].ts_event


# --- last_completed_htf: vectorised completed-bucket view for signals() overrides ----------


def test_last_completed_htf_is_nan_before_first_bucket_closes():
    df = minute_frame(90, seed=2)  # 1.5 hours of 1m bars
    htf = last_completed_htf(df, "1h")
    assert len(htf) == len(df)
    # the first hour only completes when minute 60 arrives -> rows 0..59 have no completed bar
    assert htf["close"].iloc[:60].isna().all()
    assert htf["close"].iloc[60:].notna().all()


def test_last_completed_htf_never_shows_the_in_progress_bucket():
    df = minute_frame(120, seed=4)  # exactly two hours
    htf = last_completed_htf(df, "1h")
    batch = aggregate_bars(df, "1h")
    # rows 60..119 sit inside the SECOND hour; the value they see must be the FIRST hour's
    # completed bar (strict bucket-greater-than), never the in-progress second bucket.
    first_hour_close = batch.loc[0, "close"]
    assert (htf["close"].iloc[60:120] == first_hour_close).all()
    # and never the second (in-progress) bucket's close, even on the very last base row
    assert htf["close"].iloc[119] != batch.loc[1, "close"]


def test_last_completed_htf_matches_batch_on_completed_prefix():
    df = minute_frame(150, seed=6)
    htf = last_completed_htf(df, "1h")
    batch = aggregate_bars(df, "1h")
    # at the last base row of hour 3 (row 149, inside the partial 3rd hour) the last completed
    # bar is hour 2 (batch row 1); hour 3 is still in progress.
    for col in ("open", "high", "low", "close", "volume"):
        assert htf[col].iloc[149] == pytest.approx(batch.loc[1, col])


def test_last_completed_htf_5m_to_1h_alignment():
    # feed 5m-spaced bars: a 1h bucket is 12 of them; the first completes at the 13th bar
    n = 25
    df = pd.DataFrame(
        {
            "ts_event": [i * 5 * _NS_PER_MINUTE for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [100.5 + i for i in range(n)],
            "low": [99.5 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [10.0] * n,
        }
    )
    htf = last_completed_htf(df, "1h")
    assert htf["close"].iloc[:12].isna().all()  # first hour still open
    assert htf["close"].iloc[12:].notna().all()  # 13th 5m bar opens hour 2 -> hour 1 visible
    batch = aggregate_bars(df, "1h")
    assert htf["close"].iloc[12] == pytest.approx(batch.loc[0, "close"])


# --- HTF visibility parity: streaming HtfBars vs vectorised last_completed_htf --------------
# The load-bearing property of the whole multi-timeframe surface: no completed HTF bar is ever
# visible before its bucket closes, and BOTH code paths agree on the exact base-bar index at
# which each becomes visible. Sequence-equality alone is not enough — the visibility index is
# what proves lookahead cannot creep into either path.


def _streaming_visibility(bars_list, timeframe):
    """Bar-by-bar: (base index where it becomes visible, the completed HTF Bar)."""
    from noctis.strategies.indicators import HtfBars

    htf = HtfBars(timeframe)
    events = []
    for t, bar in enumerate(bars_list):
        done = htf.add(bar)
        if done is not None:
            events.append((t, done))
    return events


def _vectorised_visibility(frame, timeframe):
    """The transitions in last_completed_htf: (base index, the newly-visible completed row)."""
    view = last_completed_htf(frame, timeframe)
    events = []
    prev_ts = None
    for t in range(len(view)):
        ts = view["ts_event"].iloc[t]
        if pd.isna(ts):
            continue
        if prev_ts is None or ts != prev_ts:
            events.append((t, view.iloc[t]))
            prev_ts = ts
    return events


def _assert_visibility_parity(frame, timeframe):
    bars_list = as_bars(frame)
    stream = _streaming_visibility(bars_list, timeframe)
    vector = _vectorised_visibility(frame, timeframe)
    assert len(stream) == len(vector), (
        f"{timeframe}: {len(stream)} streaming emissions vs {len(vector)} vectorised transitions"
    )
    for (t_s, sbar), (t_v, vrow) in zip(stream, vector, strict=True):
        assert t_s == t_v, f"{timeframe}: visible at base index {t_s} (stream) vs {t_v} (vector)"
        assert sbar.ts_event == int(vrow["ts_event"])
        assert sbar.open == pytest.approx(float(vrow["open"]))
        assert sbar.high == pytest.approx(float(vrow["high"]))
        assert sbar.low == pytest.approx(float(vrow["low"]))
        assert sbar.close == pytest.approx(float(vrow["close"]))
        assert sbar.volume == pytest.approx(float(vrow["volume"]))
    return stream


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h"])
def test_htf_visibility_parity_multiweek_continuous(timeframe):
    # ~3 weeks of continuous 1-minute bars (21 days × 1440 min) — a long stream that
    # exercises hundreds of bucket closes on both paths.
    frame = minute_frame(21 * 24 * 60, seed=17)
    events = _assert_visibility_parity(frame, timeframe)
    assert len(events) > 100  # the test actually exercised many completed buckets


def test_htf_visibility_parity_survives_gaps():
    # Halts / overnight gaps must not shift bucket boundaries or desync the two paths.
    full = minute_frame(4 * 24 * 60, seed=23)
    gapped = full.drop(index=range(500, 815)).reset_index(drop=True)  # a ~5h hole
    _assert_visibility_parity(gapped, "1h")


def test_write_gate_smoke_fixture_runs_an_htf_strategy():
    """The write-gate smoke fixture must be long enough that an HTF strategy at least runs
    through it (staying flat during warmup is legal — scenarios are the correctness oracle)."""
    from noctis.strategies.indicators import EmaState, HtfBars
    from noctis.strategies.library import fixture_frame

    frame = fixture_frame()
    htf = HtfBars("1h")
    trend = EmaState(2)
    completed = 0
    for bar in as_bars(frame):
        done = htf.add(bar)
        if done is not None:
            completed += 1
            trend.update(done)  # EmaState reads done.close — the completed HTF bar is a Bar
    assert completed >= 1, "smoke fixture too short for even one completed HTF bar"
