"""Session-context helper contract: RTH clock questions from the bar timestamp alone.

DST correctness is the module's reason to exist (vs a ``ts % 86400`` hack), so the spring
and fall transitions are tested in both directions with explicit UTC timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime

from noctis.strategies import session

_NS = 1_000_000_000


def ns(spec: str) -> int:
    """UTC-ns timestamp for 'YYYY-MM-DD HH:MM' interpreted as UTC wall time."""
    dt = datetime.strptime(spec, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    return int(dt.timestamp()) * _NS


# 2026-01-07 is a Wednesday; EST is UTC-5, so 09:30 ET == 14:30 UTC.


def test_minute_of_session_and_is_rth_boundaries_winter():
    assert session.minute_of_session(ns("2026-01-07 14:30")) == 0
    assert session.minute_of_session(ns("2026-01-07 20:59")) == 389  # 15:59 ET, last minute
    assert session.minute_of_session(ns("2026-01-07 14:29")) is None  # 09:29 ET, pre-open
    assert session.minute_of_session(ns("2026-01-07 21:00")) is None  # 16:00 ET, closed
    assert session.is_rth(ns("2026-01-07 14:30"))
    assert session.is_rth(ns("2026-01-07 20:59"))
    assert not session.is_rth(ns("2026-01-07 14:29"))
    assert not session.is_rth(ns("2026-01-07 21:00"))


def test_dst_spring_forward_week_keeps_the_open_at_0930_et():
    # Friday 2026-03-06 is EST (UTC-5): open is 14:30 UTC.
    assert session.minute_of_session(ns("2026-03-06 14:30")) == 0
    # Monday 2026-03-09 is EDT (UTC-4): open moved to 13:30 UTC...
    assert session.minute_of_session(ns("2026-03-09 13:30")) == 0
    # ...so 14:30 UTC is now an hour into the session — the modulo-86400 failure mode.
    assert session.minute_of_session(ns("2026-03-09 14:30")) == 60


def test_dst_fall_back_week_keeps_the_open_at_0930_et():
    # Friday 2026-10-30 is EDT: open at 13:30 UTC.
    assert session.minute_of_session(ns("2026-10-30 13:30")) == 0
    # Monday 2026-11-02 is EST again: 13:30 UTC is 08:30 ET, pre-open.
    assert session.minute_of_session(ns("2026-11-02 13:30")) is None
    assert session.minute_of_session(ns("2026-11-02 14:30")) == 0


def test_minutes_to_close_two_liner_semantics():
    assert session.minutes_to_close(ns("2026-01-07 14:30")) == 390  # the open
    assert session.minutes_to_close(ns("2026-01-07 20:59")) == 1  # last RTH minute
    assert session.minutes_to_close(ns("2026-01-07 21:00")) is None  # after close
    # "no entries in the last 15 minutes" flips exactly at 15:45 ET
    at_1544 = session.minutes_to_close(ns("2026-01-07 20:44"))
    at_1545 = session.minutes_to_close(ns("2026-01-07 20:45"))
    assert at_1544 is not None and at_1544 > 15
    assert at_1545 is not None and at_1545 <= 15


def test_session_date_is_the_et_trading_date():
    from datetime import date

    # 20:59 UTC on Jan 7 is still Jan 7 in ET
    assert session.session_date(ns("2026-01-07 20:59")) == date(2026, 1, 7)
    # 01:00 UTC on Jan 8 is 20:00 ET Jan 7 — the ET date, not the UTC date
    assert session.session_date(ns("2026-01-08 01:00")) == date(2026, 1, 7)


def test_new_session_edge_detector():
    monday_open = ns("2026-01-05 14:30")
    monday_noon = ns("2026-01-05 17:00")
    tuesday_open = ns("2026-01-06 14:30")
    assert session.new_session(None, monday_open)  # first bar ever
    assert not session.new_session(monday_open, monday_noon)  # same session
    assert session.new_session(monday_noon, tuesday_open)  # day rolled


def test_new_session_agrees_with_vwap_utc_day_bucketing_on_regular_sessions():
    """VwapState buckets by UTC day; for RTH bars that equals the ET session (RTH never
    crosses UTC midnight) — assert the equivalence across both DST transitions."""
    from noctis.strategies.spec.indicators import _day_ordinal

    days = ["2026-03-05", "2026-03-06", "2026-03-09", "2026-10-30", "2026-11-02", "2026-11-03"]
    opens_utc = {True: "13:30", False: "14:30"}  # EDT : EST
    stamps: list[int] = []
    for d in days:
        edt = d.startswith("2026-03-09") or d.startswith("2026-10")
        first = ns(f"{d} {opens_utc[edt]}")
        stamps.extend(first + m * 60 * _NS for m in (0, 1, 195, 389))  # open..last minute
    for prev, cur in zip(stamps, stamps[1:], strict=False):
        assert session.new_session(prev, cur) == (_day_ordinal(prev) != _day_ordinal(cur))


def test_weekend_is_not_a_session():
    # 2026-01-10/11 are Saturday/Sunday; 15:00 UTC is mid-morning ET
    assert session.minute_of_session(ns("2026-01-10 15:00")) is None
    assert session.minute_of_session(ns("2026-01-11 15:00")) is None
    assert not session.is_rth(ns("2026-01-10 15:00"))
    assert not session.is_rth(ns("2026-01-11 15:00"))
