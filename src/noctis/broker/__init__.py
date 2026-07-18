"""Noctis broker — the execution seam.

The default is an in-house paper broker (a SimulatedExchange equivalent) with fees,
slippage, positions, and P&L. The event-driven :func:`simulate` driver executes decisions
at next-bar open (no lookahead). The live adapter stays a gated stub — unreachable unless
both safety gates are open, and even then it refuses (no real-order path exists).
"""

from __future__ import annotations

from noctis.broker.live_stub import LiveBroker, LiveBrokerUnavailableError
from noctis.broker.paper import PaperBroker
from noctis.broker.persistence import AccountStore, AccountSummary
from noctis.broker.seam import (
    Broker,
    FeeModel,
    Fill,
    Order,
    OrderType,
    Position,
    Side,
    SlippageModel,
)
from noctis.broker.simulator import SimResult, simulate

__all__ = [
    "AccountStore",
    "AccountSummary",
    "Broker",
    "FeeModel",
    "Fill",
    "Order",
    "OrderType",
    "Position",
    "Side",
    "SlippageModel",
    "PaperBroker",
    "LiveBroker",
    "LiveBrokerUnavailableError",
    "SimResult",
    "simulate",
]
