"""Live bar feed backed by yfinance (Yahoo Finance) — free, delayed intraday OHLCV.

This is the live adapter of the :class:`~noctis.live.feed.BarFeed` Protocol the trading
driver (:func:`noctis.live.node.run_trading_day`) consumes — clock-bounded (never
``exhausted``; the session close ends the day). Unlike a snap-quote feed it does not build bars
from ticks — Yahoo already returns *closed* OHLCV candles at the requested ``interval`` — so
this feed pulls the recent tail on a throttled cadence and emits each newly-closed bar exactly
once, in timestamp order, keyed by symbol (one minute group per poll, so cross-symbol bars
stay aligned the way the driver's session expects).

Two disciplines are load-bearing here:

* **Never emit a forming bar.** Yahoo's most recent intraday row is the *current* (still
  open) interval and mutates on every request, so the newest row of each fetch is held back
  as *provisional* — only bars strictly older than it are committed. At a clean session close
  that tail is complete, so :meth:`flush` releases it (same role as the old feed's partial
  flush).
* **Delay is normal; staleness is not.** Yahoo intraday is delayed ~15 min — that is fine for
  paper trading and is *not* "degraded". The feed flags ``degraded`` only when a fetch fails
  or when no fresh bar has arrived for ``stale_after_s`` (Yahoo stopped advancing); the driver
  then halts order emission rather than trade on stale prices, and resumes on recovery.

Network access hides behind an injectable ``downloader`` (the default lazily imports
``yfinance``, the ``data`` extra), so the feed logic is pure and fully testable without a
network. It also **self-throttles** its fetches independently of how fast the loop polls, so a
tight poll cadence can never hammer Yahoo. Execution never goes through here — this is market
data only; paper orders route through the same broker as backtest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pandas as pd

from noctis.data.types import to_ns
from noctis.strategies.base import Bar

logger = logging.getLogger("noctis.feed")

# A downloader maps (symbols, interval) -> {symbol: OHLCV frame}. Each frame has a datetime
# index (bar-open time) and lowercase ``open/high/low/close/volume`` columns. Isolating the
# yfinance call behind this seam keeps the feed logic network-free and unit-testable.
Downloader = Callable[[list[str], str], "dict[str, pd.DataFrame]"]

# yfinance caps intraday history by interval (1m ~7d, other intraday ~60d). A small rolling
# window is all the live loop needs — enough to catch up after a mid-session restart, no more.
_PERIOD_FOR: dict[str, str] = {
    "1m": "1d",
    "2m": "5d",
    "5m": "5d",
    "15m": "5d",
    "30m": "5d",
    "60m": "5d",
    "90m": "5d",
    "1h": "5d",
}

_OHLCV = ("open", "high", "low", "close", "volume")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a yfinance frame to lowercase OHLCV columns, dropping no-trade (NaN close) rows."""
    by_lower = {str(c).lower(): c for c in df.columns}
    cols = {name: df[by_lower[name]] for name in _OHLCV if name in by_lower}
    norm = pd.DataFrame(cols)
    if "close" in norm.columns:
        norm = norm.dropna(subset=["close"])
    return norm


def _split_frames(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a (possibly multi-ticker) yfinance frame into one normalized frame per symbol."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for sym in symbols:
        if multi:
            if sym not in raw.columns.get_level_values(0):
                continue
            sub = raw[sym]
        else:  # single symbol → flat columns
            sub = raw
        norm = _normalize_columns(sub)
        if len(norm):
            out[sym] = norm
    return out


def _default_downloader(symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    """Fetch the recent intraday tail from Yahoo Finance (lazy import: only when a feed runs)."""
    import yfinance as yf  # noqa: PLC0415 — deferred so the core install needs no data extra

    period = _PERIOD_FOR.get(interval, "5d")
    raw = yf.download(
        tickers=list(symbols),
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return _split_frames(raw, symbols)


def _frame_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Convert a normalized OHLCV frame to ascending :class:`Bar` objects (UTC-ns ts_event)."""
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        close = row["close"]
        if pd.isna(close):
            continue
        vol = row.get("volume", 0.0)
        bars.append(
            Bar(
                ts_event=to_ns(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(close),
                volume=0.0 if pd.isna(vol) else float(vol),
            )
        )
    bars.sort(key=lambda b: b.ts_event)
    return bars


class YFinanceBarFeed:
    """Pull recent closed bars from yfinance and emit each newly-closed one once, in order."""

    def __init__(
        self,
        symbols: list[str],
        *,
        interval: str = "1m",
        min_interval: float = 30.0,
        stale_after_s: float = 300.0,
        downloader: Downloader | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._symbols = list(symbols)
        self._interval = interval
        self._min_interval = float(min_interval)  # min wall-seconds between real Yahoo fetches
        self._stale_after_s = float(stale_after_s)
        self._download: Downloader = downloader or _default_downloader
        self._clock = clock

        self._seen_ns: dict[str, int] = {s: -1 for s in self._symbols}  # newest committed ts
        self._provisional: dict[str, Bar] = {}  # held-back forming bar per symbol
        self._pending: dict[int, dict[str, Bar]] = {}  # ts -> {symbol: bar} awaiting a poll
        self._last_fetch: float | None = None
        self._last_bar_at: float = self._clock()  # for staleness; seeded so we don't start stale
        self._degraded = False

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def exhausted(self) -> bool:
        return False  # a live feed is clock-bounded: the session close ends the day

    def poll_once(self) -> dict[str, Bar]:
        """Fetch (if the throttle has elapsed) and return the oldest pending minute group."""
        self._maybe_fetch()
        if self._pending:
            ts = min(self._pending)
            return self._pending.pop(ts)
        return {}

    def flush(self) -> dict[str, Bar]:
        """Release the held-back tail bar per symbol — complete once the session has closed."""
        out: dict[str, Bar] = {}
        for sym, bar in self._provisional.items():
            if bar.ts_event > self._seen_ns[sym]:
                self._seen_ns[sym] = bar.ts_event
                out[sym] = bar
        return out

    # --- internals ---
    def _maybe_fetch(self) -> None:
        now = self._clock()
        if self._last_fetch is not None and (now - self._last_fetch) < self._min_interval:
            self._update_staleness(now)  # throttled poll still ages the feed toward degraded
            return
        self._last_fetch = now
        try:
            frames = self._download(self._symbols, self._interval)
        except Exception:  # noqa: BLE001 — any fetch error degrades the feed, never fails the day
            logger.warning("yfinance fetch failed; feed degraded", exc_info=True)
            self._degraded = True
            return
        if self._ingest(frames):
            self._last_bar_at = now
            self._degraded = False
        self._update_staleness(now)

    def _ingest(self, frames: dict[str, pd.DataFrame]) -> bool:
        """Queue newly-closed bars; hold each symbol's newest row back as provisional."""
        got_new = False
        for sym in self._symbols:
            df = frames.get(sym)
            if df is None or len(df) == 0:
                continue
            bars = _frame_to_bars(df)
            if not bars:
                continue
            self._provisional[sym] = bars[-1]  # newest row is the still-forming interval
            for bar in bars[:-1]:
                if bar.ts_event <= self._seen_ns[sym]:
                    continue
                self._seen_ns[sym] = bar.ts_event
                self._pending.setdefault(bar.ts_event, {})[sym] = bar
                got_new = True
        return got_new

    def _update_staleness(self, now: float) -> None:
        if now - self._last_bar_at > self._stale_after_s:
            self._degraded = True


def build_yfinance_feed(
    *,
    symbols: list[str],
    interval: str = "1m",
    min_interval: float = 30.0,
    downloader: Downloader | None = None,
) -> YFinanceBarFeed:
    """Build a :class:`YFinanceBarFeed` for ``symbols`` (no credentials, no state to cache)."""
    return YFinanceBarFeed(
        list(symbols), interval=interval, min_interval=min_interval, downloader=downloader
    )
