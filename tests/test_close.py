"""Bar-level reconciliation: live vs vendor drift on matching timestamps.

The CLOSE orchestration around it — step order, failure isolation, and the evidence the
phase finishes before it writes the day's report — lives in ``test_close_phase.py``.
"""

from __future__ import annotations

import pandas as pd

from noctis.engine.close import reconcile_bars


def _bars(closes, ts0=0):
    n = len(closes)
    return pd.DataFrame(
        {
            "ts_event": [ts0 + i * 60_000_000_000 for i in range(n)],
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [100] * n,
        }
    )


def test_reconcile_identical_bars_not_flagged():
    bars = _bars([100.0, 101.0, 102.0])
    rep = reconcile_bars(bars, bars.copy(), threshold=0.005)
    assert rep.n_compared == 3
    assert rep.max_drift == 0.0
    assert rep.flagged is False


def test_reconcile_flags_injected_drift():
    live = _bars([100.0, 101.0, 102.0])
    vendor = _bars([100.0, 101.0, 102.0])
    vendor.loc[1, ["open", "high", "low", "close"]] = [103.0, 103.1, 102.9, 103.0]  # ~2% off
    rep = reconcile_bars(live, vendor, threshold=0.005)
    assert rep.flagged is True
    assert rep.max_drift > 0.005
