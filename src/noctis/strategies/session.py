"""Session-clock helpers for authored strategies — pure functions of ``bar.ts_event``.

Answers "where am I in the trading day?" from the UTC-ns integer timestamp the ``Bar``
already carries, so ``on_bar`` code like "no entries in the last 15 minutes" or "reset my
counters at the open" stays a one- or two-liner::

    m = session.minutes_to_close(bar.ts_event)
    if m is not None and m <= 15:
        ctx.set_target(0); return

All functions assume US-equity **regular trading hours** — 09:30–16:00 America/New_York —
the same assumption the bar aggregator bakes in ("RTH never crosses UTC midnight"). DST is
handled by stdlib ``zoneinfo``: 09:30 ET is minute 0 whether that is 14:30 UTC (winter) or
13:30 UTC (summer), which is exactly what a naive ``ts % 86400`` hack gets wrong twice a
year.

**Honest limitations** (documented, deliberately not solved here):

* **No holiday calendar.** A full-holiday session simply has no bars, so holiday awareness
  buys nothing inside ``on_bar``; the exchange-calendar dependency stays behind its seam
  and out of the strategy path.
* **Half-days** (1:00 pm ET closes) are treated as full sessions, so ``minutes_to_close``
  overstates on those afternoons. A strategy that cares can watch the bar stream end.
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

__all__ = [
    "minute_of_session",
    "is_rth",
    "minutes_to_close",
    "session_date",
    "new_session",
]

_ET = ZoneInfo("America/New_York")
_NS = 1_000_000_000
_OPEN_MINUTE = 9 * 60 + 30  # 09:30 ET
_CLOSE_MINUTE = 16 * 60  # 16:00 ET (the 15:59 bar is the last RTH minute)
_SESSION_MINUTES = _CLOSE_MINUTE - _OPEN_MINUTE  # 390


def _et(ts_event: int) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(int(ts_event) // _NS, tz=_dt.UTC).astimezone(_ET)


def minute_of_session(ts_event: int) -> int | None:
    """Minutes since the 09:30 ET open: 0 at 09:30, 389 at 15:59; ``None`` outside RTH."""
    et = _et(ts_event)
    if et.weekday() >= 5:
        return None
    minute = et.hour * 60 + et.minute
    if not _OPEN_MINUTE <= minute < _CLOSE_MINUTE:
        return None
    return minute - _OPEN_MINUTE


def is_rth(ts_event: int) -> bool:
    """True when the timestamp falls inside regular trading hours (Mon–Fri 09:30–15:59 ET)."""
    return minute_of_session(ts_event) is not None


def minutes_to_close(ts_event: int) -> int | None:
    """Whole minutes until the 16:00 ET close: 390 at the open, 1 at 15:59; ``None`` outside
    RTH. Overstates on half-days (see the module docstring)."""
    m = minute_of_session(ts_event)
    return None if m is None else _SESSION_MINUTES - m


def session_date(ts_event: int) -> _dt.date:
    """The America/New_York calendar date of the timestamp — the trading date for any RTH
    bar, since a US-equity session never crosses midnight ET."""
    return _et(ts_event).date()


def new_session(prev_ts: int | None, ts_event: int) -> bool:
    """True on a session-boundary edge: the first bar ever (``prev_ts is None``) or the
    first bar whose ET trading date differs from ``prev_ts``'s. Use it to reset per-day
    state (counters, anchors, VWAP-like sums) in one condition."""
    if prev_ts is None:
        return True
    return session_date(prev_ts) != session_date(ts_event)
