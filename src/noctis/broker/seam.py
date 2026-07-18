"""The ``Broker`` seam — order submission, positions, and account P&L.

The default implementation is an in-house paper broker (a SimulatedExchange equivalent);
a live adapter stays behind the double gate. Orders, fills, and positions are plain value
types so both the paper broker and any future live/Nautilus adapter speak the same seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Provenance labels for fills — there is no resting-order book behind STOP/LIMIT."""

    MARKET = "MARKET"
    STOP = "STOP"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    ts_event: int
    reason: str = "target"  # provenance: target | stop | take_profit | trail


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0


class FeeModel:
    """Proportional commission in basis points of notional (per fill)."""

    def __init__(self, bps: float = 1.0):
        self.bps = float(bps)

    def fee(self, price: float, quantity: float) -> float:
        return abs(price * quantity) * (self.bps / 10_000.0)


class SlippageModel:
    """Proportional slippage in basis points, adverse to the order side."""

    def __init__(self, bps: float = 1.0):
        self.bps = float(bps)

    def fill_price(self, price: float, side: Side) -> float:
        adj = price * (self.bps / 10_000.0)
        return price + adj if side is Side.BUY else price - adj


@runtime_checkable
class Broker(Protocol):
    """The execution seam — the surface the trading drivers actually consume.

    Drivers speak in position *targets*, not raw orders: they mark prices as bars arrive
    (:meth:`set_price`), move positions with :meth:`rebalance_to`, and read state back for
    risk checks and reporting. An adapter that implements exactly this surface can be
    driven by the trading day.
    """

    def set_price(self, symbol: str, price: float, ts_event: int | None = None) -> None:
        """Update the mark price (drivers call this as each bar arrives)."""
        ...

    def rebalance_to(
        self, symbol: str, target_qty: float, *, price: float | None = None, reason: str = "target"
    ) -> Fill | None:
        """Order whatever delta moves the position to ``target_qty``; ``None`` if none.

        ``price=None`` fills at the current mark (today's behavior, the default everywhere);
        a value fills at that price — an exit's trigger level or the open — with slippage
        still adverse on top. ``reason`` is stamped on the fill (target | stop | take_profit
        | trail) so reporting and the forward ledger can tell exit fills apart.
        """
        ...

    def position(self, symbol: str) -> Position: ...

    def positions(self) -> dict[str, Position]: ...

    def equity(self) -> float:
        """Cash + mark-to-market value of open positions."""
        ...

    def marks(self) -> dict[str, float]:
        """The current mark prices by symbol."""
        ...

    @property
    def fills(self) -> list[Fill]:
        """Fills executed so far (session reporting slices these)."""
        ...
