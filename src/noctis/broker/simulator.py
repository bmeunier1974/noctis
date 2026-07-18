"""The event-driven simulator — the shared, lookahead-free backtest driver.

Feeds bars to a strategy's ``on_bar`` one at a time. A decision made on bar *t* (the target
position) is executed at bar *t+1*'s **open**, so the engine cannot peek at information the
strategy did not have — no lookahead, by construction. Equity is marked at each bar's close.
The same driver underpins the walk-forward validation and, later, the live paper loop.

Protective exits (the fill-model section of docs/architecture.md): when a strategy declares
``ExitRules`` with its target, the engine evaluates them intrabar between the open execution
and ``on_bar`` — open → intrabar → close, so no step sees a later step's information. After
an exit fires, the symbol latches flat until the raw target series changes value; the
recorded ``targets`` are the engine's *executed* stance (raw, suppressed to flat while
latched), which for a strategy declaring no exits is the raw series unchanged, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from noctis.broker.exits import ExitState, evaluate, ratchet
from noctis.broker.paper import PaperBroker
from noctis.broker.seam import Fill
from noctis.strategies.base import Bar, ExitRules, TargetContext, TraderStrategy


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

    rows = bars.reset_index(drop=True)
    targets: list[int] = []
    equity_curve: list[float] = []
    ts_list: list[int] = []
    pending_target = 0  # decided on the previous bar, executed at this bar's open
    pending_exits: ExitRules | None = None  # declared alongside it, enforced intrabar
    exit_state: ExitState | None = None  # anchored to the open position, None when flat
    latched = False  # an exit fired; flat until the raw target series changes value
    prev_raw_target = 0
    exits_declared = False
    exit_count = 0

    for i in range(len(rows)):
        row = rows.iloc[i]
        o = float(row["open"])
        h = float(row["high"])
        low = float(row["low"])
        c = float(row["close"])
        vol = float(row["volume"]) if "volume" in rows.columns else 0.0
        ts = int(row["ts_event"]) if "ts_event" in rows.columns else i

        # 1) execute the previous bar's decision at this bar's OPEN.
        broker.set_price(symbol, o, ts)
        prev_qty = broker.position(symbol).quantity
        fill = broker.rebalance_to(symbol, _units_for(pending_target, broker.equity(), o, alloc))
        new_qty = broker.position(symbol).quantity
        if new_qty == 0.0:
            exit_state = None  # flat clears the anchor
        elif fill is not None and (prev_qty == 0.0 or (prev_qty > 0.0) != (new_qty > 0.0)):
            # an open or a flip re-anchors exit tracking at the true entry
            exit_state = ExitState(
                direction=1 if new_qty > 0.0 else -1, entry_price=fill.price, best=fill.price
            )

        bar = Bar(ts, o, h, low, c, vol)

        # 2) intrabar: enforce the armed exit rules against this bar's range.
        if exit_state is not None and pending_exits is not None:
            trigger = evaluate(pending_exits, exit_state, bar)
            if trigger is not None:
                broker.rebalance_to(symbol, 0.0, price=trigger.price, reason=trigger.reason)
                exit_count += 1
                exit_state = None
                latched = True
            else:
                exit_state = ratchet(exit_state, bar)  # after evaluate — never before

        # 3) strategy decides for this bar (sees up to and including this close).
        strategy.on_bar(ctx, bar)
        raw_target = ctx.target
        if latched and raw_target != prev_raw_target:
            latched = False  # the strategy re-decided; the new value executes normally
        prev_raw_target = raw_target
        target = 0 if latched else raw_target
        targets.append(target)
        if ctx.exits is not None:
            exits_declared = True

        # 4) mark equity at the close.
        broker.set_price(symbol, c, ts)
        equity_curve.append(broker.equity())
        ts_list.append(ts)
        pending_target = target
        pending_exits = ctx.exits

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
