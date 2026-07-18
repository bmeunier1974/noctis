"""The market clock.

Answers ``is_open`` / ``next_open`` / ``next_close`` for the configured exchange calendar.
Holiday awareness comes from ``exchange-calendars`` when the ``data`` extra is installed;
otherwise a Monday–Friday fallback is used. Regular session hours are 09:30–16:00 in the
calendar's local timezone. Every clock read goes through an injectable ``now()`` so tests and
the smoke test can accelerate time.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

# Regular US equities session (local exchange time).
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


@cache
def _load_calendar(name: str):
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        return xcals.get_calendar(name)
    except Exception:
        return None


class MarketClock:
    """Session-aware market clock with an injectable ``now``."""

    def __init__(
        self,
        calendar: str = "XNYS",
        timezone: str = "America/New_York",
        now: Callable[[], datetime] | None = None,
    ):
        self.calendar_name = calendar
        self.tz = ZoneInfo(timezone)
        self._now = now or (lambda: datetime.now(UTC))
        self._cal = _load_calendar(calendar)

    # --- time source ---
    def now(self) -> datetime:
        t = self._now()
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        return t.astimezone(UTC)

    def _at(self, at: datetime | None) -> datetime:
        if at is None:
            return self.now()
        return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)

    # --- session model ---
    def _is_session_day(self, d: date) -> bool:
        if self._cal is not None:
            try:
                import pandas as pd

                return bool(self._cal.is_session(pd.Timestamp(d)))
            except Exception:
                pass
        return d.weekday() < 5

    def _session_bounds(self, d: date) -> tuple[datetime, datetime]:
        open_local = datetime.combine(d, SESSION_OPEN, tzinfo=self.tz)
        close_local = datetime.combine(d, SESSION_CLOSE, tzinfo=self.tz)
        return open_local.astimezone(UTC), close_local.astimezone(UTC)

    # --- public API ---
    def is_open(self, at: datetime | None = None) -> bool:
        moment = self._at(at)
        local_day = moment.astimezone(self.tz).date()
        if not self._is_session_day(local_day):
            return False
        open_utc, close_utc = self._session_bounds(local_day)
        return open_utc <= moment < close_utc

    def next_open(self, at: datetime | None = None) -> datetime:
        moment = self._at(at)
        start_day = moment.astimezone(self.tz).date()
        for offset in range(0, 400):
            d = start_day + timedelta(days=offset)
            if self._is_session_day(d):
                open_utc, _ = self._session_bounds(d)
                if open_utc > moment:
                    return open_utc
        raise RuntimeError("no session open found within a year")

    def next_close(self, at: datetime | None = None) -> datetime:
        moment = self._at(at)
        start_day = moment.astimezone(self.tz).date()
        for offset in range(0, 400):
            d = start_day + timedelta(days=offset)
            if self._is_session_day(d):
                _, close_utc = self._session_bounds(d)
                if close_utc > moment:
                    return close_utc
        raise RuntimeError("no session close found within a year")

    def has_holiday_awareness(self) -> bool:
        return self._cal is not None
