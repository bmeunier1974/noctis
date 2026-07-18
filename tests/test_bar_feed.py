"""The bar-feed seam: the BarFeed Protocol and its catalog adapter (ReplayBarFeed).

The trading driver consumes one contract — ``symbols`` / ``degraded`` / ``exhausted`` /
``poll_once`` / ``flush`` — with two adapters: the live yfinance feed (clock-bounded, tested
in ``test_yfinance_feed.py``) and the catalog replay (data-bounded, tested here). These pin
the replay adapter's semantics: timestamp-ordered minute groups with cross-symbol alignment,
exhaustion when the timeline drains, an empty flush (catalog bars are complete), and the
injectable degraded flag. Plus the one runtime-level regression the seam exists to prevent:
a **live** session must advance the session high-water mark exactly like a replay session,
so a later replay day can never re-trade a live-traded one on the carried account.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from noctis.data.types import NS_PER_SECOND
from noctis.engine.sessions import SessionLedger
from noctis.live import BarFeed, ReplayBarFeed

from ._data_helpers import make_ohlcv


def _shifted(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = df.copy()
    out["ts_event"] = out["ts_event"] + minutes * 60 * NS_PER_SECOND
    return out


# --- the catalog adapter -------------------------------------------------------------------


def test_replay_feed_satisfies_the_barfeed_protocol():
    feed = ReplayBarFeed({"AAPL": make_ohlcv([100.0, 101.0])})
    assert isinstance(feed, BarFeed)


def test_replay_feed_polls_ordered_aligned_minute_groups():
    aapl = make_ohlcv([100.0, 101.0, 102.0])
    msft = _shifted(make_ohlcv([200.0, 201.0, 202.0]), 1)  # overlaps minutes 1–2, adds 3
    feed = ReplayBarFeed({"AAPL": aapl, "MSFT": msft})
    assert feed.symbols == ["AAPL", "MSFT"]

    groups = []
    while not feed.exhausted:
        groups.append(feed.poll_once())
    assert [sorted(g) for g in groups] == [
        ["AAPL"],  # minute 0: AAPL only
        ["AAPL", "MSFT"],  # minutes 1–2: the shared timestamps arrive as one group each
        ["AAPL", "MSFT"],
        ["MSFT"],  # minute 3: MSFT's extra bar
    ]
    ts = [next(iter(g.values())).ts_event for g in groups]
    assert ts == sorted(ts)  # strictly timestamp-ordered
    shared = groups[1]
    assert shared["AAPL"].ts_event == shared["MSFT"].ts_event  # cross-symbol alignment


def test_replay_feed_exhausts_then_polls_empty_and_flushes_nothing():
    feed = ReplayBarFeed({"AAPL": make_ohlcv([100.0])})
    assert not feed.exhausted
    assert feed.poll_once()  # the one group
    assert feed.exhausted  # data-bounded: a drained timeline ends the day
    assert feed.poll_once() == {}
    assert feed.flush() == {}  # catalog bars are complete; nothing is ever held back


def test_replay_feed_drops_empty_symbols_and_defers_degraded_to_the_callable():
    flag = {"degraded": False}
    feed = ReplayBarFeed(
        {"AAPL": make_ohlcv([100.0]), "EMPTY": make_ohlcv([])},
        degraded=lambda: flag["degraded"],
    )
    assert feed.symbols == ["AAPL"]  # a symbol with no bars that day never trades
    assert feed.degraded is False
    flag["degraded"] = True
    assert feed.degraded is True  # per-poll re-evaluation, like the old is_degraded callable
    assert ReplayBarFeed({"AAPL": make_ohlcv([100.0])}).degraded is False  # default: healthy


# --- the regression the one settle path exists to prevent ----------------------------------


def test_live_session_advances_the_high_water_mark_like_replay(tmp_path):
    """Before the TradingDay settle unification the live path saved the account but never
    the session ledger, so a live-traded day followed by a replay day (e.g. after a feed
    outage) was silently re-traded on the carried account. One settle path for both
    drivers is what closes that gap — pin it at the runtime level."""
    from zoneinfo import ZoneInfo

    from tests.test_live_feed import (
        _groups_from_bars,
        _make_settings,
        _RecordingFeedFactory,
        _run_one_cycle,
        _seed_catalog,
        _uptrend,
    )

    settings, lake_dir = _make_settings(tmp_path, provider="yfinance")
    lake = _seed_catalog(lake_dir)
    factory = _RecordingFeedFactory(_groups_from_bars(_uptrend(4), "AAPL"))
    runtime, _result = _run_one_cycle(settings, lake, feed_factory=factory, tmp_path=tmp_path)

    assert factory.calls >= 1  # the live driver genuinely ran (not the replay fallback)
    ledger_path = tmp_path / "state" / "trading_sessions.json"
    traded = SessionLedger(ledger_path).load()
    assert traded is not None  # the live session advanced the high-water mark…
    # …to the session date the cycle traded (the run starts 2027-01-04 06:00 ET).
    assert traded == datetime(2027, 1, 4, 6, 0, tzinfo=ZoneInfo("America/New_York")).date()
