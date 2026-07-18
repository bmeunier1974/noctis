"""Session identity + the trading high-water mark (the rolling live-holdout).

A *session* is an exchange-calendar trading day: a bar belongs to session ``d`` when its
``ts_event`` (int64 UTC nanoseconds) converted to the exchange timezone falls on local date
``d``. Grouping by local date handles half-days for free — whatever bars the lake has for
that date are the session. The lake syncs T+1, so any date present is complete by
construction.

The TRADING phase trades only sessions newer than the persisted high-water mark
(``state/trading_sessions.json``), so each day's replay is exactly the slice of bars no
tuning ever saw — never the whole catalog.
"""

from __future__ import annotations

import json
from datetime import date, tzinfo
from pathlib import Path

import pandas as pd

_LEDGER_VERSION = 1


def session_date(ts_event_ns: int, tz: tzinfo) -> date:
    """Exchange-local calendar date for a UTC-nanosecond bar timestamp."""
    return pd.Timestamp(int(ts_event_ns), unit="ns", tz="UTC").tz_convert(tz).date()


def _local_dates(df: pd.DataFrame, tz: tzinfo) -> pd.Series:
    """Each bar's exchange-local calendar date (vectorised ``session_date``)."""
    return pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(tz).dt.date


def sessions_present(bars_by_symbol: dict[str, pd.DataFrame], tz: tzinfo) -> list[date]:
    """Sorted unique session dates present across the roster's bars."""
    present: set[date] = set()
    for df in bars_by_symbol.values():
        if len(df) > 0:
            present.update(_local_dates(df, tz).unique())
    return sorted(present)


def slice_session(
    bars_by_symbol: dict[str, pd.DataFrame], d: date, tz: tzinfo
) -> dict[str, pd.DataFrame]:
    """Each symbol's bars whose exchange-local date is ``d``.

    Symbols with no bars that day yield empty frames (the batch trading driver already
    drops empties).
    """
    out: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        if len(df) == 0:
            out[sym] = df.copy()
            continue
        mask = (_local_dates(df, tz) == d).to_numpy()
        out[sym] = df.loc[mask].reset_index(drop=True)
    return out


def unseen_sessions(
    present: list[date], last_traded: date | None, cap: int
) -> tuple[list[date], int]:
    """The session dates to trade (chronological) plus the count truncated by ``cap``.

    ``last_traded=None`` (first run ever) → only the newest present session, never all of
    history — this single rule is what kills the full-catalog replay. Otherwise every
    session in ``(last_traded, newest]`` is unseen; when there are more than ``cap``, only
    the newest ``cap`` are kept (still chronological) and the rest are reported as skipped.
    """
    if not present:
        return [], 0
    if last_traded is None:
        return [max(present)], 0
    unseen = sorted(d for d in present if d > last_traded)
    truncated = 0
    if cap > 0 and len(unseen) > cap:
        truncated = len(unseen) - cap
        unseen = unseen[-cap:]
    return unseen, truncated


class SessionLedger:
    """The trading high-water mark: the last session date already traded.

    Persisted with atomic writes (tmp file + ``replace``, like the champion registry) and
    written only **after** a session's replay completes — a crash mid-replay re-trades that
    session rather than silently skipping it. An absent file means "never traded" (the
    first run trades only the newest session). A corrupt file is a hard error, not a silent
    reset — silently resetting would re-trade history.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> date | None:
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text())
            return date.fromisoformat(data["last_traded_session"])
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"corrupt trading-session ledger {self.path}: {exc!r} — refusing to "
                "silently reset (that would re-trade history); repair or delete the file"
            ) from exc

    def save(self, last_traded: date) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": _LEDGER_VERSION, "last_traded_session": last_traded.isoformat()}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic on POSIX
