"""The paper broker — an in-house SimulatedExchange.

Market orders fill immediately at the current mark price adjusted for slippage, with a
proportional commission. Cash, positions, and realised/unrealised P&L are tracked so the
event backtest and the (later) live paper loop share one honest fills model. No real-money
order path exists here — that is the live adapter's job, and it is gated shut.
"""

from __future__ import annotations

from noctis.broker.seam import (
    FeeModel,
    Fill,
    Order,
    Position,
    Side,
    SlippageModel,
)


class PaperBroker:
    """Simulated exchange: fills, slippage, fees, positions, P&L."""

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
    ):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.fee_model = fee_model or FeeModel()
        self.slippage_model = slippage_model or SlippageModel()
        # symbol -> [quantity, avg_price]
        self._positions: dict[str, list[float]] = {}
        self._marks: dict[str, float] = {}
        self._fills: list[Fill] = []
        self.realized_pnl = 0.0
        # Realized P&L split per symbol (sums to ``realized_pnl``), for per-champion forward
        # attribution (live-holdout plan 5). Fees are NOT deducted here — they live in
        # ``fees_paid`` — so the invariant ``sum(by_symbol.values()) == realized_pnl`` holds.
        self.realized_pnl_by_symbol: dict[str, float] = {}
        self.fees_paid = 0.0
        self.slippage_cost = 0.0
        self._ts = 0

    # --- price marks (driver updates these as bars arrive) ---
    def set_price(self, symbol: str, price: float, ts_event: int | None = None) -> None:
        self._marks[symbol] = float(price)
        if ts_event is not None:
            self._ts = int(ts_event)

    def marks(self) -> dict[str, float]:
        return dict(self._marks)

    # --- Broker seam ---
    def submit_order(
        self, order: Order, *, price: float | None = None, reason: str = "target"
    ) -> Fill:
        if order.quantity <= 0:
            raise ValueError("order quantity must be positive")
        base = float(price) if price is not None else self._marks.get(order.symbol)
        if base is None:
            raise RuntimeError(f"no mark price for {order.symbol}; call set_price first")
        fill_price = self.slippage_model.fill_price(base, order.side)
        fee = self.fee_model.fee(fill_price, order.quantity)
        self.slippage_cost += abs(fill_price - base) * order.quantity
        self._apply_fill(order.symbol, order.side, order.quantity, fill_price, fee)
        fill = Fill(order.symbol, order.side, order.quantity, fill_price, fee, self._ts, reason)
        self._fills.append(fill)
        return fill

    def _apply_fill(self, symbol: str, side: Side, qty: float, price: float, fee: float) -> None:
        signed = qty if side is Side.BUY else -qty
        pos = self._positions.setdefault(symbol, [0.0, 0.0])
        cur_qty, avg = pos
        new_qty = cur_qty + signed

        # cash: buying spends, selling receives; fees always cost.
        self.cash -= signed * price
        self.cash -= fee
        self.fees_paid += fee

        if cur_qty == 0 or (cur_qty > 0) == (signed > 0):
            # opening or increasing in the same direction → weighted avg price
            total = abs(cur_qty) + abs(signed)
            avg = (abs(cur_qty) * avg + abs(signed) * price) / total if total else 0.0
        else:
            # reducing / closing / flipping → realise P&L on the closed portion
            closed = min(abs(signed), abs(cur_qty))
            direction = 1.0 if cur_qty > 0 else -1.0
            pnl = direction * closed * (price - avg)
            self.realized_pnl += pnl
            self.realized_pnl_by_symbol[symbol] = self.realized_pnl_by_symbol.get(symbol, 0.0) + pnl
            if abs(signed) > abs(cur_qty):
                # flipped through zero → remainder opens at fill price
                avg = price
            # if fully/partly closed but not flipped, avg unchanged
        if new_qty == 0:
            avg = 0.0
        pos[0], pos[1] = new_qty, avg

    def position(self, symbol: str) -> Position:
        qty, avg = self._positions.get(symbol, [0.0, 0.0])
        return Position(symbol, qty, avg)

    def positions(self) -> dict[str, Position]:
        return {s: self.position(s) for s, (q, _a) in self._positions.items() if q != 0.0}

    def equity(self) -> float:
        value = self.cash
        for symbol, (qty, _avg) in self._positions.items():
            if qty != 0.0:
                mark = self._marks.get(symbol)
                if mark is not None:
                    value += qty * mark
        return value

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    # --- persistence (the continuous paper account carried across sessions) ---
    def to_dict(self) -> dict:
        """Serializable account state: cash, open positions, marks, and cost totals.

        Marks are kept for open positions so ``equity()`` is computable straight after a
        reload — that is the next session's daily-loss anchor before its first bar. Fills
        are per-session report material, not account state: persisting them would grow the
        file forever and double-count trades in later sessions' reports.
        """
        positions = {
            sym: {"qty": qty, "avg_price": avg}
            for sym, (qty, avg) in self._positions.items()
            if qty != 0.0
        }
        return {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "positions": positions,
            "marks": {sym: self._marks[sym] for sym in positions if sym in self._marks},
            "realized_pnl": self.realized_pnl,
            "realized_pnl_by_symbol": dict(self.realized_pnl_by_symbol),
            "fees_paid": self.fees_paid,
            "slippage_cost": self.slippage_cost,
            "last_ts": self._ts,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> PaperBroker:
        """Rebuild a broker from :meth:`to_dict` output.

        Fee/slippage models are behaviour, not state — they come from config, exactly as
        for a fresh broker. Malformed data raises (KeyError/TypeError/ValueError); the
        caller decides what a corrupt account file means.
        """
        broker = cls(
            float(data["starting_cash"]), fee_model=fee_model, slippage_model=slippage_model
        )
        broker.cash = float(data["cash"])
        broker._positions = {
            str(sym): [float(p["qty"]), float(p["avg_price"])]
            for sym, p in data["positions"].items()
        }
        broker._marks = {str(sym): float(v) for sym, v in data["marks"].items()}
        broker.realized_pnl = float(data["realized_pnl"])
        # Missing on older account files (pre plan 5) → empty, so they load unchanged.
        broker.realized_pnl_by_symbol = {
            str(sym): float(v) for sym, v in data.get("realized_pnl_by_symbol", {}).items()
        }
        broker.fees_paid = float(data["fees_paid"])
        broker.slippage_cost = float(data["slippage_cost"])
        broker._ts = int(data["last_ts"])
        return broker

    # --- helper used by the event driver ---
    def rebalance_to(
        self, symbol: str, target_qty: float, *, price: float | None = None, reason: str = "target"
    ) -> Fill | None:
        """Submit a market order to move the position to ``target_qty``. No-op if equal.

        ``price=None`` fills at the current mark (the default everywhere); a value fills at
        that price — an exit's trigger level or the open — with slippage still adverse on top.
        ``reason`` lands on the fill so reporting can tell exit fills apart.
        """
        current = self.position(symbol).quantity
        delta = target_qty - current
        if abs(delta) < 1e-9:
            return None
        side = Side.BUY if delta > 0 else Side.SELL
        return self.submit_order(Order(symbol, side, abs(delta)), price=price, reason=reason)
