"""The position driver — the one module that owns the fill-model step order.

One symbol, one position, walked through bars under the fill model the whole system is
built on: **decide on bar t, fill at bar t+1's open**. A bar is two calls in a fixed
order — :meth:`PositionDriver.at_open` (the broker-touching half: mark the open, execute
the target decided on the previous bar) then :meth:`PositionDriver.at_close` (the deciding
half: ``on_bar``, carry the new target and its rules forward, mark the close). Calling
them out of order raises rather than silently skipping a step, because a step order that
can drift is exactly the bug this module exists to prevent.

The backtest simulator and the live trading day are its two drivers. They differ only in
what they pass in — never in the step order, which lives here alone. It talks to the
:class:`~noctis.broker.seam.Broker` seam (mark, position, rebalance) and not to any one
implementation of it. **Sizing is a seam** too (:class:`Sizer`: how many units a target
means at this price): the backtest's allocation formula and the live risk manager are its
two adapters, and a dead-band skip or a risk refusal is the adapter's answer (``None``),
not the driver's. The whole protective-exit step lives here too — anchored at the fill price on
every open or flip, evaluated intrabar before the strategy decides, ratcheted only when
nothing fired, and latched flat until the strategy asks for something new — so the exit
engine (:mod:`noctis.broker.exits`) has exactly one caller per bar and its conservative
policy is never re-implemented.

The driver returns what happened and nothing more: it never logs, counts, or emits
events — that is the caller's job, and keeping it out is what lets both drivers share
this code without dragging their reporting along.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from noctis.broker.exits import ExitState, ExitTrigger, evaluate, ratchet
from noctis.broker.seam import Broker, Fill
from noctis.strategies.base import Bar, ExitRules, TargetContext, TraderStrategy

__all__ = ["CloseOutcome", "OpenOutcome", "PositionDriver", "Sizer"]


class Sizer(Protocol):
    """How many units a target means at this price. ``None`` = skip this bar.

    A ``None`` answer is a refusal to trade *this* bar (the live dead-band, a risk
    limit), not a change of mind: the pending target stays pending and the next bar
    asks again.
    """

    def __call__(self, symbol: str, target: int, price: float) -> float | None: ...


@dataclass(frozen=True)
class OpenOutcome:
    """What the open half of a bar did."""

    fill: Fill | None
    skipped: bool  # the sizer answered None — pending held, nothing submitted


@dataclass(frozen=True)
class CloseOutcome:
    """What the deciding half of a bar did."""

    exit_fill: Fill | None
    trigger: ExitTrigger | None
    target: int  # the executed stance (flat while latched)
    raw_target: int  # what the strategy actually asked for


class PositionDriver:
    """One symbol's position, walked bar by bar under the fill model."""

    def __init__(
        self,
        broker: Broker,
        symbol: str,
        strategy: TraderStrategy,
        ctx: TargetContext,
        sizer: Sizer,
        *,
        pending_target: int = 0,
        exit_state: ExitState | None = None,
    ) -> None:
        self._broker = broker
        self._symbol = symbol
        self._strategy = strategy
        self._ctx = ctx
        self._sizer = sizer
        self._pending_target = int(pending_target)
        self._pending_exits: ExitRules | None = None
        self._prev_raw_target = int(pending_target)
        self._latched = False
        self._opened = False
        # Exit tracking for the open position, re-anchored on every open or flip; None
        # while flat. Carried here so the exit engine has one caller per bar.
        self.exit_state = exit_state

    @classmethod
    def from_position(
        cls,
        broker: Broker,
        symbol: str,
        strategy: TraderStrategy,
        ctx: TargetContext,
        sizer: Sizer,
    ) -> PositionDriver:
        """Seed a driver from whatever position the account already carries.

        A carried position (the continuous paper account across sessions) keeps its
        direction until the strategy first decides — seeding flat would force-flatten it
        at the first bar, injecting turnover nobody chose — and anchors its exit tracking
        at its true entry, the account's average price. A flat account seeds flat with no
        anchor, so a fresh run behaves exactly as if it had just started.

        ``on_start`` is deliberately *not* called here: the caller owns the strategy's
        lifecycle (both drivers start their strategies before the first bar).
        """
        pos = broker.position(symbol)
        qty = pos.quantity
        return cls(
            broker,
            symbol,
            strategy,
            ctx,
            sizer,
            pending_target=(qty > 0.0) - (qty < 0.0),
            exit_state=(
                ExitState(
                    direction=1 if qty > 0.0 else -1,
                    entry_price=pos.avg_price,
                    best=pos.avg_price,
                )
                if qty != 0.0
                else None
            ),
        )

    # --- read-only carried state ---
    @property
    def pending_target(self) -> int:
        """The stance decided on the previous bar, executed at the next open."""
        return self._pending_target

    @property
    def pending_exits(self) -> ExitRules | None:
        """The protective rules declared alongside the pending target."""
        return self._pending_exits

    @property
    def latched(self) -> bool:
        """True while an exit holds the symbol flat, until the raw target changes."""
        return self._latched

    # --- the two halves of a bar, in this order ---
    def at_open(self, bar: Bar, *, execute: bool = True) -> OpenOutcome:
        """Mark the open, then execute the target decided on the previous bar.

        ``execute=False`` marks only (a halted or degraded session still needs an honest
        mark), leaving the pending target to be re-offered at the next open.
        """
        self._broker.set_price(self._symbol, bar.open, bar.ts_event)
        self._opened = True
        if not execute:
            return OpenOutcome(fill=None, skipped=False)

        units = self._sizer(self._symbol, self._pending_target, bar.open)
        if units is None:
            return OpenOutcome(fill=None, skipped=True)

        prev_qty = self._broker.position(self._symbol).quantity
        fill = self._broker.rebalance_to(self._symbol, units)
        new_qty = self._broker.position(self._symbol).quantity
        if new_qty == 0.0:
            self.exit_state = None  # flat clears the anchor
        elif fill is not None and (prev_qty == 0.0 or (prev_qty > 0.0) != (new_qty > 0.0)):
            # an open or a flip re-anchors exit tracking at the true entry
            self.exit_state = ExitState(
                direction=1 if new_qty > 0.0 else -1, entry_price=fill.price, best=fill.price
            )
        return OpenOutcome(fill=fill, skipped=False)

    def at_close(self, bar: Bar, *, execute: bool = True) -> CloseOutcome:
        """Enforce the armed exits intrabar, then let the strategy decide for this bar.

        The rules evaluated here are the ones carried from the *previous* close, against
        this bar's full range — a position opened at this bar's open is protected from
        this bar on. A trigger flattens at the trigger price, clears the anchor and
        latches; only when nothing fires does the favorable extreme ratchet, so the trail
        is always measured from the prior bar's extreme (see :mod:`noctis.broker.exits`).

        ``execute=False`` suppresses order emission — and with it the whole exit step,
        ratchet included, so a suppressed bar cannot advance a trail it was never allowed
        to act on. The strategy still sees the bar and the close is still marked, so a
        suppressed bar leaves no hole in the series the next decision is made from.
        """
        if not self._opened:
            raise RuntimeError(
                f"at_open must run before at_close for {self._symbol}: a decision that is "
                "not preceded by its bar's open execution would break the fill model"
            )
        self._opened = False

        exit_fill: Fill | None = None
        trigger: ExitTrigger | None = None
        if execute and self.exit_state is not None and self._pending_exits is not None:
            trigger = evaluate(self._pending_exits, self.exit_state, bar)
            if trigger is not None:
                exit_fill = self._broker.rebalance_to(
                    self._symbol, 0.0, price=trigger.price, reason=trigger.reason
                )
                self.exit_state = None  # flat clears the anchor
                self._latched = True
            else:
                self.exit_state = ratchet(self.exit_state, bar)  # after evaluate — never before

        self._strategy.on_bar(self._ctx, bar)
        raw_target = self._ctx.target
        if self._latched and raw_target != self._prev_raw_target:
            self._latched = False  # the strategy re-decided; the new value executes normally
        self._prev_raw_target = raw_target
        target = 0 if self._latched else raw_target
        self._pending_target = target
        self._pending_exits = self._ctx.exits

        self._broker.set_price(self._symbol, bar.close, bar.ts_event)
        return CloseOutcome(
            exit_fill=exit_fill, trigger=trigger, target=target, raw_target=raw_target
        )
