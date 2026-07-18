"""Author-facing HtfBars: a higher-timeframe aggregator owned by a strategy.

HtfBars wraps the streaming aggregator (a bucket is emitted only when the first bar of the
next bucket arrives), so a strategy consulting a 1h trend filter from 1m bars stays
lookahead-free by construction. These tests pin the delivery contract and the alignment
guard; the streaming-vs-vectorised visibility parity is a separate test (#60).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noctis.strategies import indicators as ind
from noctis.strategies.base import Bar

_NS_PER_MINUTE = 60 * 1_000_000_000


def _minute_bars(n: int, start_ts: int = 0) -> list[Bar]:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    return [
        Bar(
            start_ts + i * _NS_PER_MINUTE,
            float(close[i]),
            float(close[i] + 0.5),
            float(close[i] - 0.5),
            float(close[i]),
            float(rng.integers(100, 1000)),
        )
        for i in range(n)
    ]


def test_htf_emits_completed_bucket_when_next_opens():
    htf = ind.HtfBars("1h")
    bars = _minute_bars(120)  # two full hours of 1m bars
    completed = [htf.add(b) for b in bars]
    # the first 1h bar is emitted only when minute 60 (the next bucket's first bar) arrives
    assert all(c is None for c in completed[:60])
    assert completed[60] is not None
    first_hour = completed[60]
    assert first_hour.open == bars[0].open
    assert first_hour.close == bars[59].close
    assert first_hour.high == max(b.high for b in bars[:60])
    assert first_hour.low == min(b.low for b in bars[:60])
    assert first_hour.ts_event == bars[59].ts_event  # stamped with its LAST constituent
    # nothing else completes until the third hour would open
    assert all(c is None for c in completed[61:])


def test_htf_never_emits_a_partial_final_bucket():
    htf = ind.HtfBars("1h")
    bars = _minute_bars(90)  # one full hour + a 30-minute partial
    completed = [c for c in (htf.add(b) for b in bars) if c is not None]
    assert len(completed) == 1  # only the completed first hour; the partial is never emitted


def test_htf_agrees_with_batch_aggregation_on_completed_buckets():
    from noctis.data.aggregate import aggregate_bars

    bars = _minute_bars(150)
    htf = ind.HtfBars("1h")
    streamed = [c for c in (htf.add(b) for b in bars) if c is not None]
    frame = pd.DataFrame(
        {
            "ts_event": [b.ts_event for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    batch = aggregate_bars(frame, "1h")
    # batch keeps the trailing partial; streamed drops it → compare on the completed prefix
    assert len(streamed) == len(batch) - 1
    for i, sbar in enumerate(streamed):
        assert sbar.ts_event == batch.loc[i, "ts_event"]
        assert sbar.close == batch.loc[i, "close"]
        assert sbar.high == batch.loc[i, "high"]


def test_htf_rejects_1m_passthrough():
    with pytest.raises(ValueError, match="1m"):
        ind.HtfBars("1m")


def test_htf_rejects_unsupported_timeframe():
    with pytest.raises(ValueError):
        ind.HtfBars("banana")
