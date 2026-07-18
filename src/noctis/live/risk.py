"""Risk limits enforced before every order in the TRADING loop.

* ``max_position_pct`` — a single symbol's position notional cannot exceed this % of equity.
* ``max_gross_exposure_pct`` — the sum of |position notional| across symbols is capped.
* ``max_daily_loss_pct`` — once equity falls this far below the session start, new
  exposure-increasing orders are refused for the rest of the session (flattening is allowed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 10.0
    max_gross_exposure_pct: float = 100.0
    max_daily_loss_pct: float = 3.0


@dataclass(frozen=True)
class RiskDecision:
    target_qty: float
    refused: bool
    reason: str


class RiskManager:
    """Turns a desired directional signal into a risk-clamped target quantity."""

    def __init__(self, limits: RiskLimits, start_equity: float):
        self.limits = limits
        self.start_equity = float(start_equity)
        self._halted = False

    def is_halted(self, equity: float) -> bool:
        # Latches: once the daily-loss floor is breached, the halt sticks for the rest of
        # the session even if equity recovers. A manager is built per *session date* — the
        # replay driver slices bars to one session per _TradingSession, never spanning days,
        # so a latch can refuse at most one session's orders.
        floor = self.start_equity * (1.0 - self.limits.max_daily_loss_pct / 100.0)
        if equity <= floor:
            self._halted = True
        return self._halted

    def target(
        self,
        symbol: str,
        desired_sign: int,
        price: float,
        equity: float,
        positions: dict[str, float],
        marks: dict[str, float],
    ) -> RiskDecision:
        """Clamp a desired position (sign in {-1,0,1}) to the risk limits."""
        current = positions.get(symbol, 0.0)

        if desired_sign == 0 or price <= 0:
            return RiskDecision(0.0, refused=False, reason="flat target")

        # Per-position notional cap.
        max_pos_notional = equity * self.limits.max_position_pct / 100.0
        # Gross exposure room (exclude this symbol's current contribution).
        gross_used_other = sum(
            abs(q * marks.get(s, 0.0)) for s, q in positions.items() if s != symbol
        )
        gross_cap = equity * self.limits.max_gross_exposure_pct / 100.0
        gross_room = max(0.0, gross_cap - gross_used_other)

        notional = min(max_pos_notional, gross_room)
        target_qty = desired_sign * notional / price

        refused = False
        reason = "within limits"
        if abs(target_qty) < abs(current):
            reason = "reduced by exposure cap"
        if self.is_halted(equity) and abs(target_qty) > abs(current):
            # Daily loss breached: forbid increasing exposure; hold current (or allow exits).
            return RiskDecision(
                current, refused=True, reason="daily loss limit breached — new exposure refused"
            )
        if notional <= 0:
            refused = True
            reason = "no exposure room"
        return RiskDecision(target_qty, refused=refused, reason=reason)
