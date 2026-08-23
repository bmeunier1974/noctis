"""Noctis engine — the loops and the market-clock state machine."""

from __future__ import annotations

from noctis.engine.clock import MarketClock
from noctis.engine.close import ClosePhase, CloseResult, ReconciliationReport, reconcile_bars
from noctis.engine.forward_ledger import (
    ForwardLedger,
    ForwardRecord,
    champion_key,
    forward_records,
)
from noctis.engine.machine import Phase, TradingMachine, initial_phase_for
from noctis.engine.pacing import (
    BoundedWaiter,
    RealSleeper,
    SimulatedSleeper,
    Sleeper,
    StallGuard,
    StopFlag,
)
from noctis.engine.report_assembly import SessionActivity, assemble_report
from noctis.engine.research import ResearchSummary, run_research
from noctis.engine.runtime import Runtime, RuntimeResult, build_runtime
from noctis.engine.trading_phase import (
    SessionRecord,
    TradingOutcome,
    TradingPhase,
    resolve_trading_driver,
)

__all__ = [
    "MarketClock",
    "ClosePhase",
    "CloseResult",
    "ReconciliationReport",
    "reconcile_bars",
    "Phase",
    "TradingMachine",
    "initial_phase_for",
    "Sleeper",
    "RealSleeper",
    "SimulatedSleeper",
    "BoundedWaiter",
    "StallGuard",
    "StopFlag",
    "ResearchSummary",
    "run_research",
    "SessionActivity",
    "assemble_report",
    "Runtime",
    "RuntimeResult",
    "build_runtime",
    "TradingPhase",
    "TradingOutcome",
    "SessionRecord",
    "resolve_trading_driver",
    "ForwardLedger",
    "ForwardRecord",
    "champion_key",
    "forward_records",
]
