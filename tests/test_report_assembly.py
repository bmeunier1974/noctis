"""assemble_report — one home for the close-of-day report wiring.

The pure render (test_reporting) and the CLOSE orchestration (test_close) were already
covered; these tests pin the *assembly*: persisted state (registry, account, forward
ledger, specs, memory) and one session's activity land in the right ReportData fields.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.broker.paper import PaperBroker
from noctis.broker.persistence import AccountStore
from noctis.champions import ChampionRegistry, PromotionRules
from noctis.engine.forward_ledger import ForwardLedger
from noctis.engine.report_assembly import SessionActivity, assemble_report
from noctis.memory.base import InMemoryMemory
from noctis.reporting.report import Trade, render_report

RULES = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)


def _metrics(sharpe: float) -> Metrics:
    return Metrics(
        total_return=0.0,
        sharpe=sharpe,
        sortino=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        turnover=0.0,
        exposure=0.0,
    )


def _scorecard(family: str, test: float, train: float, **params) -> Scorecard:
    return Scorecard(
        family=family,
        params=params,
        metric_name="sharpe",
        stage="validated",
        symbols={"FIT": SymbolScore(splits=[SplitScore(0, _metrics(train), _metrics(test))])},
    )


def _registry_with_champion(state_dir, family: str, **params) -> ChampionRegistry:
    reg = ChampionRegistry(state_dir / "champions.json", capacity=3)
    assert reg.consider(_scorecard(family, 1.5, 1.7, **params), RULES).promote
    return reg


def test_assemble_from_persisted_state_alone(tmp_path):
    """No session passed: every populated field comes from persisted state; session fields
    are the honest zeros (this is exactly what `noctis report` generates outside a run)."""
    state = tmp_path / "state"
    reg = _registry_with_champion(state, "sma_crossover", fast=5, slow=20)
    AccountStore(state / "paper_account.json").save(PaperBroker(), date(2026, 1, 2))
    ledger = ForwardLedger(state / "forward_ledger.json")
    ledger.record("sma_crossover|abc", "sma_crossover", date(2026, 1, 5), {"AAPL": 12.5})
    ledger.save()
    memory = InMemoryMemory()
    memory.append_finding("PROMOTED sma_crossover")

    data = assemble_report(
        as_of="2026-01-06", mode="paper", registry=reg, memory=memory, state_dir=state
    )

    assert data.as_of == "2026-01-06" and data.mode == "paper"
    assert data.champions == [
        {
            "family": "sma_crossover",
            "params": {"fast": 5, "slow": 20},
            "test_metric": pytest.approx(1.5),
            "gap": pytest.approx(0.2),
        }
    ]
    assert [h["family"] for h in data.promotions] == ["sma_crossover"]
    assert data.demotions == []
    assert data.cumulative_pnl == pytest.approx(0.0)
    assert data.account_opened == "2026-01-02"
    assert len(data.forward) == 1
    assert data.forward[0]["family"] == "sma_crossover"
    assert data.forward[0]["forward_pnl"] == pytest.approx(12.5)
    assert data.research["findings"] == ["PROMOTED sma_crossover"]
    # No session: equity/trades/positions/events/counters are all zero-valued.
    assert data.start_equity == 0.0 and data.realized_pnl == 0.0
    assert data.trades == [] and data.positions == {} and data.events == []
    assert data.research["iterations"] == 0 and data.research["minted"] == []


def test_assemble_folds_session_activity(tmp_path):
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    session = SessionActivity(start_equity=100_000.0, end_equity=100_250.0)
    session.trades.append(Trade("AAPL", "buy", 10, 190.0, "champion signal"))
    session.positions["AAPL"] = 10.0
    session.research_iterations = 7
    session.research_promotions = 1
    session.minted_specs.append("spec_x")
    session.events.append("2 orders refused by risk limits")

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
        session=session,
    )

    assert data.start_equity == 100_000.0 and data.end_equity == 100_250.0
    assert data.realized_pnl == pytest.approx(250.0)
    assert data.trades == session.trades and data.trades is not session.trades
    assert data.positions == {"AAPL": 10.0}
    assert data.research["iterations"] == 7 and data.research["promotions"] == 1
    assert data.research["minted"] == ["spec_x"]
    assert data.events == ["2 orders refused by risk limits"]
    assert data.events is not session.events  # run_close appends to the report's own copy
    # The assembled data renders end-to-end (the render is the report contract).
    assert "Close-of-day report — 2026-01-06" in render_report(data)


def test_assemble_folds_undecided_strategies(tmp_path):
    """A session's undecided strategies (authored but never carried to a verdict) pass
    through beside the research counters, as a copy the report owns."""
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    session = SessionActivity()
    session.research_undecided.extend(["draft_a", "draft_b"])

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
        session=session,
    )

    assert data.research["undecided"] == ["draft_a", "draft_b"]
    assert data.research["undecided"] is not session.research_undecided


def test_assemble_empty_undecided_is_an_empty_entry(tmp_path):
    """A session that left nothing unresolved still carries the key — an empty list, not a
    missing entry — so consumers (JSON, QA rollups) read one shape."""
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
    )

    assert data.research["undecided"] == []


def _ledgered_session(state_dir, session_id: str = "session-x"):
    """Write a real two-candidate SessionLedger arc (one escalated author, one reject, one
    approve+promote) and return it — the ledger the CLOSE report reads a rollup + trail from."""
    from noctis.research.ledger import SessionLedger

    led = SessionLedger(state_dir, session_id)
    led.record_session_start(mandate="m", budgets={}, models={"driver": "d"})
    led.record_thesis("momo_1", "buy strength")
    led.record_stage("formulate")
    led.record_episode(stage="formulate", model="driver", tokens=12, outcome="ok")
    led.record_stage("match", strategy="momo_1")
    led.record_stage("author", strategy="momo_1")
    led.record_stage("optimize", strategy="momo_1", detail={"trials": 5, "best_metric": 1.2})
    led.record_stage("decide", strategy="momo_1")
    led.record_episode(stage="decide", model="driver", tokens=8, outcome="ok")
    led.record_verdict("momo_1", verdict="reject", lesson="thin", promoted=False)
    led.record_thesis("rev_2", "fade the spike")
    led.record_stage("author", strategy="rev_2")
    led.record_episode(stage="author", model="coder-paid", tokens=40, outcome="ok", escalated=True)
    led.record_stage("optimize", strategy="rev_2", detail={"trials": 7, "best_metric": 2.5})
    led.record_stage("decide", strategy="rev_2")
    led.record_verdict("rev_2", verdict="approve", lesson="edge holds", promoted=True)
    led.record_session_end(formulated=2, promoted=1, rejected=1, note="max_episodes")
    return led


def test_assemble_threads_the_ledger_rollup_and_candidate_trail(tmp_path):
    """A folded episodic summary carrying a ledger path lands a per-session rollup + candidate
    trail in the research block, derived from the session ledger."""
    from noctis.reporting.report import render_report

    state = tmp_path / "state"
    led = _ledgered_session(state, "session-x")
    reg = ChampionRegistry(state / "champions.json", capacity=3)
    session = SessionActivity()
    session.research_ledgers.append(str(led.path))

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=state,
        session=session,
    )

    sessions = data.research["sessions"]
    assert len(sessions) == 1
    rollup = sessions[0]["rollup"]
    assert rollup["theses"] == 2 and rollup["authored"] == 2
    assert rollup["trials"] == 12 and rollup["escalations"] == 1
    assert rollup["verdicts"] == {"approve": 1, "reject": 1}
    assert [c["strategy"] for c in sessions[0]["candidates"]] == ["momo_1", "rev_2"]
    # It renders end to end.
    text = render_report(data)
    assert "Theses formulated: 2" in text and "momo_1" in text


def test_assemble_without_a_ledger_adds_no_sessions_key(tmp_path):
    """No ledger path folded ⇒ the research block carries no ``sessions`` key at all, so a
    ledgerless (conversation-loop / legacy / `noctis report`) render is byte-identical to today."""
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
    )
    assert "sessions" not in data.research


def test_assemble_tolerates_a_missing_or_malformed_ledger(tmp_path):
    """A folded ledger path that points at a missing/empty file never breaks the report — that
    session simply contributes no rollup (graceful degradation to today's rendering)."""
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    session = SessionActivity()
    session.research_ledgers.append(str(tmp_path / "state" / "sessions" / "ghost.jsonl"))

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
        session=session,
    )
    assert "sessions" not in data.research  # the missing ledger contributed nothing


def test_corrupt_account_omits_curve_and_keeps_forward_realized(tmp_path):
    """An unreadable paper account degrades to no cumulative line — never an error — and
    the forward section falls back to realized-only (no broker to mark unrealized)."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "paper_account.json").write_text("{not json")
    ledger = ForwardLedger(state / "forward_ledger.json")
    ledger.record("rsi_meanrev|k", "rsi_meanrev", date(2026, 1, 5), {"MSFT": -3.0})
    ledger.save()

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=ChampionRegistry(state / "champions.json", capacity=3),
        memory=InMemoryMemory(),
        state_dir=state,
    )

    assert data.cumulative_pnl is None and data.account_opened is None
    assert data.forward[0]["realized_pnl"] == pytest.approx(-3.0)
    assert data.forward[0]["unrealized_pnl"] == 0.0


def test_minted_spec_champions_are_flagged(tmp_path):
    """A champion whose family is a persisted spec shows up in research.promoted_specs;
    seed-family champions do not."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "specs.json").write_text(json.dumps({"version": 1, "specs": {"spec_momo": {}}}))
    reg = _registry_with_champion(state, "spec_momo")
    assert reg.consider(_scorecard("donchian_breakout", 1.2, 1.3), RULES).promote

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=state,
    )

    assert data.research["promoted_specs"] == ["spec_momo"]
    assert {c["family"] for c in data.champions} == {"spec_momo", "donchian_breakout"}
