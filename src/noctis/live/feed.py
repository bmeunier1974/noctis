"""The bar-feed seam: what a TRADING session drinks its minutes from.

:class:`BarFeed` is the one contract the trading driver consumes — a source of *completed*
minute groups, polled one group at a time in timestamp order. Two adapters satisfy it:

* :class:`ReplayBarFeed` (here) — a static catalog slice as a feed. **Data-bounded**: it is
  ``exhausted`` once its timeline is drained, never degraded on its own, and holds nothing
  back (``flush`` is empty — catalog bars are complete by construction).
* :class:`~noctis.data.yfinance.feed.YFinanceBarFeed` — the live adapter. **Clock-bounded**:
  never ``exhausted`` (the session close ends the day), ``degraded`` on fetch failure or
  staleness, and its still-forming tail bar is released only by ``flush`` at a clean close.

Whatever the adapter, execution never happens here — a feed is market data only; orders
route through the paper broker in the driver.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import pandas as pd

from noctis.strategies.base import Bar


@runtime_checkable
class BarFeed(Protocol):
    """A source of completed bars, polled one cross-symbol minute group at a time.

    The driver's whole vocabulary: ``symbols`` names who may trade, ``poll_once`` yields the
    oldest pending minute group (``{}`` when nothing is ready *yet*), ``exhausted`` says the
    feed can never yield again (a drained replay; a live feed is bounded by the clock
    instead), ``degraded`` halts order emission while observation continues, and ``flush``
    releases any held-back tail once the session is over.
    """

    @property
    def symbols(self) -> list[str]: ...

    @property
    def degraded(self) -> bool: ...

    @property
    def exhausted(self) -> bool: ...

    def poll_once(self) -> dict[str, Bar]: ...

    def flush(self) -> dict[str, Bar]: ...


def _row_to_bar(row) -> Bar:
    return Bar(
        ts_event=int(row["ts_event"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]) if "volume" in row else 0.0,
    )


class ReplayBarFeed:
    """A static timeline (a catalog session slice, a backtest tape) as a :class:`BarFeed`.

    Each ``poll_once`` pops the oldest minute across symbols as one group, so cross-symbol
    bars stay aligned exactly the way the streaming adapters deliver them — replay and live
    differ only in where the minutes come from. ``degraded`` defers to an optional callable
    (the runtime's feed-health flag); the default is a healthy feed.
    """

    def __init__(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        *,
        degraded: Callable[[], bool] | None = None,
    ):
        # Retained as provenance: the exact timeline this feed replays.
        self.bars_by_symbol = bars_by_symbol
        self._degraded_fn = degraded
        by_ts: dict[int, dict[str, Bar]] = {}
        for sym, df in bars_by_symbol.items():
            for _, row in df.reset_index(drop=True).iterrows():
                bar = _row_to_bar(row)
                by_ts.setdefault(bar.ts_event, {})[sym] = bar
        self._groups: list[dict[str, Bar]] = [by_ts[ts] for ts in sorted(by_ts)]
        self._i = 0

    @property
    def symbols(self) -> list[str]:
        return sorted(s for s, df in self.bars_by_symbol.items() if len(df) > 0)

    @property
    def degraded(self) -> bool:
        return bool(self._degraded_fn()) if self._degraded_fn is not None else False

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._groups)

    def poll_once(self) -> dict[str, Bar]:
        if self.exhausted:
            return {}
        group = self._groups[self._i]
        self._i += 1
        return group

    def flush(self) -> dict[str, Bar]:
        return {}  # catalog bars are complete; nothing is ever held back
