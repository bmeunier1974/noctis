"""Trading-calendar abstraction.

Uses ``exchange-calendars`` when installed (the ``data`` extra); otherwise falls back to a
Monday–Friday weekday calendar so the integrity check and tests run without the heavy
dependency. The fallback is a deliberate approximation — real holidays are only honored
when ``exchange-calendars`` is present.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache


@cache
def _load_xcal(name: str):
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        return xcals.get_calendar(name)
    except Exception:
        return None


def trading_sessions(start: date, end: date, calendar: str = "XNYS") -> list[date]:
    """Trading session dates in ``[start, end]`` inclusive for ``calendar``."""
    if end < start:
        return []
    cal = _load_xcal(calendar)
    if cal is not None:
        import pandas as pd

        sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [ts.date() for ts in sessions]
    # Weekday fallback (Mon–Fri), no holiday awareness.
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def has_holiday_awareness(calendar: str = "XNYS") -> bool:
    """True when a real exchange calendar (not the weekday fallback) is in use."""
    return _load_xcal(calendar) is not None
