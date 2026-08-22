"""The backtest driver — a frame of bars in, executed targets and an equity curve out.

One of the two drivers of :class:`noctis.broker.position_driver.PositionDriver`, which owns
the fill-model step order the whole system rests on: a decision made on bar *t* (the target
position) is executed at bar *t+1*'s **open**, protective exits are enforced intrabar between
that execution and ``on_bar``, and equity is marked at the close — so no step ever sees a
later step's information and there is no lookahead, by construction. The live trading day
drives the same driver over the same steps, which is why the two cannot disagree about a fill.

This module only walks the frame and collects what came back: the executed targets, the marked
equity curve, the fills. The recorded ``targets`` are the engine's *executed* stance (the raw
target, suppressed to flat while an exit latch holds), which for a strategy declaring no exits
is the raw series unchanged, byte for byte; ``_extra["exit_count"]`` is reported only when the
strategy declared ``ExitRules`` at least once. Sizing is the driver's :class:`Sizer` seam, and
this module's adapter is the ``alloc`` fraction of marked equity at the execution price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from noctis.broker.paper import PaperBroker
from noctis.broker.position_driver import PositionDriver, Sizer
from noctis.broker.seam import Fill
from noctis.strategies.base import Bar, TargetContext, TraderStrategy


@dataclass
class SimResult:
    targets: list[int]
    equity_curve: list[float]
    timestamps: list[int]
    fills: list[Fill]
    final_equity: float
    starting_equity: float
    symbol: str = "SYM"
    _extra: dict = field(default_factory=dict)

    def equity_series(self) -> pd.Series:
        return pd.Series(self.equity_curve, index=pd.Index(self.timestamps, name="ts_event"))

    def returns(self) -> pd.Series:
        return self.equity_series().pct_change().fillna(0.0)


def _units_for(target: int, equity: float, price: float, alloc: float) -> float:
    if target == 0 or price <= 0:
        return 0.0
    return target * (alloc * equity) / price


def _alloc_sizer(broker: PaperBroker, alloc: float) -> Sizer:
    """The backtest's sizing adapter: ``alloc`` of the marked equity behind a target.

    Sized at the execution price off the equity the broker has marked at that moment, so a
    growing account scales its stake. It never refuses a bar — a risk-driven ``None`` is the
    live adapter's answer, not the backtest's.
    """

    def size(symbol: str, target: int, price: float) -> float | None:
        return _units_for(target, broker.equity(), price, alloc)

    return size


def simulate(
    strategy: TraderStrategy,
    bars: pd.DataFrame,
    broker: PaperBroker | None = None,
    symbol: str = "SYM",
    alloc: float = 0.95,
) -> SimResult:
    """Run ``strategy`` over ``bars`` with next-bar-open execution."""
    broker = broker or PaperBroker()
    starting_equity = broker.equity()
    ctx = TargetContext()
    strategy.on_start(ctx)
    sizer = _alloc_sizer(broker, alloc)
    driver = PositionDriver.from_position(broker, symbol, strategy, ctx, sizer)

    rows = bars.reset_index(drop=True)
    has_volume = "volume" in rows.columns
    has_ts = "ts_event" in rows.columns
    targets: list[int] = []
    equity_curve: list[float] = []
    ts_list: list[int] = []
    exits_declared = False
    exit_count = 0

    for i in range(len(rows)):
        row = rows.iloc[i]
        ts = int(row["ts_event"]) if has_ts else i
        bar = Bar(
            ts,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]) if has_volume else 0.0,
        )
        driver.at_open(bar)
        outcome = driver.at_close(bar)  # ends by marking the close: equity below is that mark
        targets.append(outcome.target)
        if outcome.trigger is not None:
            exit_count += 1  # every trigger, fill or not — an exit that moved nothing still fired
        if ctx.exits is not None:
            exits_declared = True
        equity_curve.append(broker.equity())
        ts_list.append(ts)

    return SimResult(
        targets=targets,
        equity_curve=equity_curve,
        timestamps=ts_list,
        fills=broker.fills,
        final_equity=broker.equity(),
        starting_equity=starting_equity,
        symbol=symbol,
        _extra={"exit_count": exit_count} if exits_declared else {},
    )
