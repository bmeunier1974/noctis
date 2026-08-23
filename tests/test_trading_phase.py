"""One trading module: the trading day IS the trading phase (epic #264, story #269).

A TRADING entry is one cohesive job — assemble the session collaborators, trade each session,
settle it (the crash-safe order: account first, high-water mark second), and fold what it did
into one :class:`~noctis.engine.TradingOutcome`. That job now lives in one module,
:mod:`noctis.engine.trading_phase`: :class:`~noctis.engine.TradingDay` settles a session
straight into the outcome the phase is building and hands back the stamped
:class:`~noctis.live.node.TradingSummary` that is the session's evidence, so there is no
per-session wrapper shape between the settle and the fold.

The other half of the merge is the fill costs. ``backtest.fee_bps``/``backtest.slippage_bps``
are what the broker charges *and* what a trade row reports; two reads of one setting are two
chances to disagree, so the phase resolves them **once** and hands the same numbers to the
account it loads and to the day that stamps the rows. The last test here is the guard that
keeps it one read.

Session-level replay behaviour (slicing, catch-up, account continuity, attribution) lives in
``tests/test_session_slice.py``, ``tests/test_account_continuity.py`` and
``tests/test_forward_ledger.py``; the driver-choice matrix lives in
``tests/test_runtime_trading.py``.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import noctis.engine as engine
import noctis.engine.trading_phase as trading_phase
from noctis.broker import FeeModel, PaperBroker, SlippageModel
from noctis.broker.persistence import AccountStore
from noctis.engine.forward_ledger import ForwardLedger
from noctis.engine.sessions import SessionLedger
from noctis.engine.trading_phase import TradingDay, TradingOutcome
from noctis.live import RiskLimits
from noctis.live.feed import ReplayBarFeed
from noctis.strategies import FamilyRegistry
from noctis.strategies.base import TraderStrategy

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "noctis"

SESSION = date(2026, 7, 6)
FEE_BPS, SLIPPAGE_BPS = 2.5, 3.0  # deliberately not the shipped 1.0/1.0 defaults


@dataclass(frozen=True)
class _NoParams:
    pass


class _AlwaysLong(TraderStrategy):
    """Buys on its first bar and holds — the smallest thing that fills."""

    name = "always_long"
    params_cls = _NoParams
    timeframe = "1m"

    def on_start(self, ctx):
        pass

    def on_bar(self, ctx, bar):
        ctx.set_target(1)

    @classmethod
    def param_space(cls):
        return []


class _Entry:
    """A champion entry stand-in with just what the slots and attribution read."""

    family = "always_long"
    params: dict = {}
    crowned_at = "t1"
    live_symbols = ["S"]
    test_metric = 1.0


class _Registry:
    def list(self):
        return [_Entry()]


def _tape(closes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event": [i * 60 * 1_000_000_000 for i in range(len(closes))],
            "open": [float(c) for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": [float(c) for c in closes],
            "volume": [1000.0] * len(closes),
        }
    )


def _trading_day(tmp_path) -> TradingDay:
    """One TradingDay on a real account/ledger/forward tier under ``tmp_path``."""
    families = FamilyRegistry()
    families.register(_AlwaysLong)
    forward = ForwardLedger(tmp_path / "forward_ledger.json")
    forward.load()
    return TradingDay(
        broker=PaperBroker(
            starting_cash=100_000.0,
            fee_model=FeeModel(FEE_BPS),
            slippage_model=SlippageModel(SLIPPAGE_BPS),
        ),
        store=AccountStore(tmp_path / "paper_account.json"),
        ledger=SessionLedger(tmp_path / "trading_sessions.json"),
        forward=forward,
        registry=_Registry(),
        families=families,
        limits=RiskLimits(
            max_position_pct=95.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=100.0
        ),
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )


# ── one module ──────────────────────────────────────────────────────────────────────────────
def test_the_trading_day_lives_in_the_trading_phase_module():
    """``noctis.engine.trading_day`` is gone: the day, the fold and the phase are one module,
    and ``SessionOutcome`` — the per-session wrapper the fold-into-the-outcome move emptied —
    exists nowhere in the package."""
    assert importlib.util.find_spec("noctis.engine.trading_day") is None
    assert engine.TradingDay is trading_phase.TradingDay
    leftovers = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "SessionOutcome" in path.read_text()
    ]
    assert leftovers == []


# ── the settle folds into the outcome ───────────────────────────────────────────────────────
def test_run_folds_the_session_into_the_outcome_and_returns_its_stamped_summary(tmp_path):
    """One settled session lands in the outcome the phase is building: the stamped summary on
    ``sessions``, its fills as trade rows, its report lines, its orders/equity/positions, and
    the bars it recorded. The return value is that same summary — the session's evidence."""
    day = _trading_day(tmp_path)
    outcome = TradingOutcome()

    summary = day.run(
        ReplayBarFeed({"S": _tape([100.0, 101.0, 102.0, 103.0])}),
        SESSION,
        outcome=outcome,
        record_bars=True,
    )

    assert outcome.sessions == [summary]
    assert summary.session == SESSION  # the settle stamps the date it speaks for
    assert summary.fills > 0
    assert [t.symbol for t in outcome.trades] == ["S"] * summary.fills
    assert all(t.rationale == "champion signal" for t in outcome.trades)
    assert all(t.champion == "always_long" for t in outcome.trades)
    assert all(t.ts is not None for t in outcome.trades)
    assert outcome.orders_submitted == summary.orders_submitted > 0
    assert outcome.positions == summary.positions
    assert outcome.start_equity == summary.start_equity
    assert outcome.end_equity == summary.final_equity
    assert outcome.events == summary.events
    assert list(outcome.live_bars) == ["S"]  # record_bars: the session's own bars come back
    # …and the settle order is unchanged: account first, high-water mark second.
    assert (tmp_path / "paper_account.json").is_file()
    assert SessionLedger(tmp_path / "trading_sessions.json").load() == SESSION


def test_a_catch_up_folds_each_session_into_the_same_outcome(tmp_path):
    """Two sessions on one carried account accumulate into one outcome: both summaries in
    traded order, every fill reported exactly once, the scalars from the last session."""
    day = _trading_day(tmp_path)
    outcome = TradingOutcome()

    first = day.run(ReplayBarFeed({"S": _tape([100.0, 101.0, 102.0])}), SESSION, outcome=outcome)
    second = day.run(
        ReplayBarFeed({"S": _tape([120.0, 121.0, 122.0])}),
        date(2026, 7, 7),
        outcome=outcome,
    )

    assert outcome.sessions == [first, second]
    assert [s.session for s in outcome.sessions] == [SESSION, date(2026, 7, 7)]
    # The carried broker accumulates fills across the catch-up; each is a row exactly once.
    assert len(outcome.trades) == first.fills + second.fills
    assert outcome.end_equity == second.final_equity
    assert outcome.positions == second.positions


# ── the fill costs are one number, read once ────────────────────────────────────────────────
def test_every_trade_row_reports_the_cost_its_fill_was_charged(tmp_path):
    """The row's reported per-side costs are the ones the account filled under: the day stamps
    its own fee/slippage, never a second read of the setting that built the broker."""
    day = _trading_day(tmp_path)
    outcome = TradingOutcome()

    day.run(ReplayBarFeed({"S": _tape([100.0, 101.0, 102.0])}), SESSION, outcome=outcome)

    assert outcome.trades
    assert all(t.slippage_bps == SLIPPAGE_BPS for t in outcome.trades)
    charged = [t.fees / (t.quantity * t.price) * 10_000.0 for t in outcome.trades]
    assert charged == pytest.approx([FEE_BPS] * len(outcome.trades))


def _cost_setting_reads(tree: ast.AST) -> list[str]:
    """Every ``…backtest.fee_bps`` / ``…backtest.slippage_bps`` read in a parsed module."""
    return [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in ("fee_bps", "slippage_bps")
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "backtest"
    ]


def test_the_configured_fill_costs_are_read_once_in_the_trading_phase():
    """Exactly one read of each cost setting in the whole engine package, both in
    ``TradingPhase.__init__``. A second read is a second chance for the broker's cost and the
    reported cost to disagree — which is the bug this shape makes unrepresentable."""
    reads = {
        str(path.relative_to(REPO_ROOT)): _cost_setting_reads(ast.parse(path.read_text()))
        for path in sorted((SOURCE_ROOT / "engine").rglob("*.py"))
    }
    assert {file: names for file, names in reads.items() if names} == {
        "src/noctis/engine/trading_phase.py": ["fee_bps", "slippage_bps"]
    }

    module = ast.parse((SOURCE_ROOT / "engine" / "trading_phase.py").read_text())
    init = next(
        node
        for cls in ast.walk(module)
        if isinstance(cls, ast.ClassDef) and cls.name == "TradingPhase"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert _cost_setting_reads(init) == ["fee_bps", "slippage_bps"]
