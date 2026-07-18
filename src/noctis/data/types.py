"""Shared types and time helpers for the data lake.

Timestamps are UTC **nanoseconds since the epoch** (int64) end-to-end, matching the
event-engine convention. Bars are stored as pandas DataFrames with a fixed column set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

# A bare ISO calendar date with no time component, e.g. "2024-01-10".
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The US-equity trading calendar's timezone — vendor T+1 boundaries follow it.
US_MARKET_TZ = ZoneInfo("America/New_York")

NS_PER_SECOND = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_SECOND

# Canonical bar schema. ts_event is int64 UTC ns; prices float64; volume int64.
BAR_COLUMNS: tuple[str, ...] = ("ts_event", "open", "high", "low", "close", "volume")
_BAR_DTYPES = {
    "ts_event": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}


@dataclass(frozen=True)
class SeriesKey:
    """Identifies one time series in the lake: (dataset, schema, symbol)."""

    dataset: str
    schema: str
    symbol: str

    @property
    def slug(self) -> str:
        return f"{self.dataset}/{self.schema}/{self.symbol}"

    @property
    def rel_path(self) -> str:
        return f"{self.dataset}/{self.schema}/{self.symbol}.parquet"


def empty_bars() -> pd.DataFrame:
    """An empty bar frame with the canonical schema and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=_BAR_DTYPES[col]) for col in BAR_COLUMNS})
    return df


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a bar frame to the canonical schema: columns, dtypes, sorted, unique ts."""
    if df is None or len(df) == 0:
        return empty_bars()
    out = df.loc[:, list(BAR_COLUMNS)].copy()
    for col, dtype in _BAR_DTYPES.items():
        out[col] = out[col].astype(dtype)
    out = out.drop_duplicates(subset="ts_event", keep="last")
    out = out.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    return out


def to_ns(value) -> int:
    """Convert a date/datetime/ISO-string/Timestamp/int to UTC nanoseconds."""
    if isinstance(value, int):
        return int(value)
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(value.year, value.month, value.day, tz="UTC")
    else:  # str or other parseable
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value)


def to_ns_end_inclusive(value) -> int:
    """End-inclusive ns for an ``--end`` boundary.

    A *date-only* value (a plain :class:`date`, or an ISO string like ``"2024-01-10"`` with
    no time component) means "through the end of that day", so this returns the last ns of
    the day via :func:`day_bounds_ns`. Anything carrying an explicit time — a ``datetime``,
    a ``Timestamp``, an int, or an ISO string with a time part — defers to :func:`to_ns`
    unchanged. This is used only for parsing the inclusive ``--end`` of a date range; the
    plain :func:`to_ns` still governs coverage boundaries.
    """
    if isinstance(value, datetime):  # datetime is a date subclass — check it first
        return to_ns(value)
    if isinstance(value, date):
        return day_bounds_ns(value)[1]
    if isinstance(value, str) and _DATE_ONLY_RE.match(value.strip()):
        return day_bounds_ns(pd.Timestamp(value.strip()).date())[1]
    return to_ns(value)


def ns_to_timestamp(ns: int) -> pd.Timestamp:
    """UTC Timestamp for an int64 nanosecond value."""
    return pd.Timestamp(int(ns), unit="ns", tz="UTC")


def ns_to_date(ns: int) -> date:
    """UTC calendar date for an int64 nanosecond value."""
    return ns_to_timestamp(ns).date()


def day_start_ns(d: date) -> int:
    """UTC-midnight nanoseconds for a calendar date."""
    return to_ns(datetime(d.year, d.month, d.day, tzinfo=UTC))


def t1_boundary_ns(now: datetime) -> int:
    """T+1 vendor boundary: UTC midnight of *now*'s US-market (America/New_York) date.

    Vendor availability for US-equity datasets follows the **ET** trading day — DataBento's
    historical tier ends at the current session's ET midnight (04:00Z/05:00Z), and an end
    past it is a 403 license error, not just a 422. The date must be taken in ET, not UTC:
    between 8 PM ET and midnight ET the UTC calendar has already rolled to tomorrow, and
    UTC-midnight of *that* date lands a day past the license line. Flooring to the ET date
    loses no bars — US equities print nothing between the 8 PM ET extended close and
    midnight ET.
    """
    return day_start_ns(now.astimezone(US_MARKET_TZ).date())


def day_bounds_ns(d: date) -> tuple[int, int]:
    """[start, end] nanoseconds spanning a single UTC calendar day (end inclusive)."""
    start = day_start_ns(d)
    return start, start + NS_PER_DAY - 1
