"""assemble_report — one home for the close-of-day report wiring.

The pure render (test_reporting) and the CLOSE orchestration (test_close) were already
covered; these tests pin the *assembly*: persisted state (registry, account, forward
ledger, specs, memory) and one session's activity land in the right ReportData fields.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.broker.paper import PaperBroker
from noctis.broker.persistence import AccountStore, AccountSummary
from noctis.broker.seam import Order, Side
from noctis.champions import ChampionRegistry, PromotionRules
from noctis.engine.forward_ledger import ForwardLedger, champion_key
from noctis.engine.report_assembly import (
    AccountForward,
    SessionActivity,
    assemble_report,
    gather_account_forward,
)
from noctis.engine.research import ResearchSummary
from noctis.engine.trading_phase import TradingOutcome
from noctis.live.node import TradingSummary
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


def _scorecard(
    family: str, test: float, train: float, *, symbol: str = "FIT", **params
) -> Scorecard:
    return Scorecard(
        family=family,
        params=params,
        metric_name="sharpe",
        stage="validated",
        symbols={symbol: SymbolScore(splits=[SplitScore(0, _metrics(train), _metrics(test))])},
    )


def _registry_with_champion(
    state_dir, family: str, *, symbol: str = "FIT", **params
) -> ChampionRegistry:
    """A board with one champion. ``symbol`` is the name it was tuned on — and therefore the
    one an open position in the account is attributed to when forward records are built."""
    reg = ChampionRegistry(state_dir / "champions.json", capacity=3)
    assert reg.consider(_scorecard(family, 1.5, 1.7, symbol=symbol, **params), RULES).promote
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
    assert data.champions == (
        {
            "family": "sma_crossover",
            "params": {"fast": 5, "slow": 20},
            "test_metric": pytest.approx(1.5),
            "gap": pytest.approx(0.2),
        },
    )
    assert [h["family"] for h in data.promotions] == ["sma_crossover"]
    assert data.demotions == ()
    assert data.cumulative_pnl == pytest.approx(0.0)
    assert data.account_opened == "2026-01-02"
    assert len(data.forward) == 1
    assert data.forward[0]["family"] == "sma_crossover"
    assert data.forward[0]["forward_pnl"] == pytest.approx(12.5)
    assert data.research["findings"] == ["PROMOTED sma_crossover"]
    # No session: equity/trades/positions/events/counters are all zero-valued.
    assert data.start_equity == 0.0 and data.realized_pnl == 0.0
    assert data.trades == () and data.positions == {} and data.events == ()
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
    assert list(data.trades) == session.trades  # a frozen copy the report owns
    assert data.positions == {"AAPL": 10.0}
    assert data.research["iterations"] == 7 and data.research["promotions"] == 1
    assert data.research["minted"] == ["spec_x"]
    assert data.events == ("2 orders refused by risk limits",)
    # The report's own frozen copy: the cycle can go on accumulating events, the assembled
    # report cannot change under whoever wrote it.
    session.events.append("a later event the report never saw")
    assert data.events == ("2 orders refused by risk limits",)
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


def _account_and_ledger(state, entries=()) -> None:
    """A persisted account holding 10 AAPL marked at +30/share, plus a forward ledger entry —
    keyed to the champion the open position is attributed to when one is passed, so the record
    carries both halves (ledger realized + account unrealized)."""
    broker = PaperBroker()
    broker.set_price("AAPL", 100.0)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    broker.set_price("AAPL", 130.0)
    AccountStore(state / "paper_account.json").save(broker, date(2026, 1, 2))
    key = champion_key(entries[0]) if entries else "sma_crossover|abc"
    ledger = ForwardLedger(state / "forward_ledger.json")
    ledger.record(key, "sma_crossover", date(2026, 1, 5), {"AAPL": 12.5})
    ledger.save()


def test_gather_account_forward_parses_the_account_once(tmp_path, monkeypatch):
    """One `load()` feeds both the account summary and the forward ledger's unrealized marks —
    one parse of `paper_account.json` where there were two (story #266)."""
    state = tmp_path / "state"
    reg = _registry_with_champion(state, "sma_crossover", symbol="AAPL", fast=5, slow=20)
    _account_and_ledger(state, reg.list())
    loads: list[str] = []
    real_load = AccountStore.load

    def counting_load(self, *args, **kwargs):
        loads.append(str(self.path))
        return real_load(self, *args, **kwargs)

    monkeypatch.setattr(AccountStore, "load", counting_load)

    af = gather_account_forward(state, reg.list())

    assert len(loads) == 1  # one parse, where the summary and the broker were two
    assert not af.account_corrupt
    assert af.account is not None
    assert af.account.opened == "2026-01-02"
    assert af.account.open_positions == 1
    # That one load fed the forward ledger too: the open position is marked against it.
    assert af.records[0].unrealized_pnl == pytest.approx(300.0, abs=1.0)
    assert af.records[0].realized_pnl == pytest.approx(12.5)


def test_assemble_report_reads_the_same_account_forward_it_can_be_given(tmp_path):
    """`account_forward=None` means "read it yourself" (the `noctis report` path); given one,
    it is used as-is — and for one account file both produce the same report."""
    state = tmp_path / "state"
    reg = _registry_with_champion(state, "sma_crossover", symbol="AAPL", fast=5, slow=20)
    _account_and_ledger(state, reg.list())
    kwargs = dict(
        as_of="2026-01-06",
        mode="paper",
        registry=reg,
        memory=InMemoryMemory(),
        state_dir=state,
    )

    read_it_itself = assemble_report(**kwargs)
    handed_one = assemble_report(
        **kwargs, account_forward=gather_account_forward(state, reg.list())
    )

    assert handed_one == read_it_itself
    # The account really was read: 10 AAPL bought at 100 and marked at 130, minus fees.
    assert handed_one.cumulative_pnl == pytest.approx(300.0, abs=1.0)
    assert handed_one.account_opened == "2026-01-02"
    assert handed_one.forward[0]["unrealized_pnl"] == pytest.approx(300.0, abs=1.0)


def test_assemble_report_uses_the_account_forward_it_is_given(tmp_path):
    """A given AccountForward is the account the report states — never re-read behind it, so
    CLOSE's one read is the one the written report carries."""
    state = tmp_path / "state"
    _account_and_ledger(state)
    given = AccountForward(
        account=AccountSummary(
            equity=123_456.0,
            starting_cash=100_000.0,
            cumulative_pnl=23_456.0,
            open_positions=2,
            opened="2025-12-01",
            last_session="2026-01-05",
        ),
        account_corrupt=False,
        forward=ForwardLedger(state / "forward_ledger.json"),
        records=[],
    )

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=ChampionRegistry(state / "champions.json", capacity=3),
        memory=InMemoryMemory(),
        state_dir=state,
        account_forward=given,
    )

    assert data.cumulative_pnl == pytest.approx(23_456.0)
    assert data.account_opened == "2025-12-01"
    assert data.forward == ()  # the given view's records, not the file's


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


# ── SessionActivity folds the cycle's own evidence ───────────────────────────────────────
# The cycle's accounting sits beside the cycle's shape (epic #264, story #270): the RESEARCH
# and TRADING phases hand the accumulator their result and it folds itself, instead of the
# runtime loop copying field by field.


def test_fold_research_copies_the_summary_into_the_cycle():
    """One research session's summary lands in the cycle verbatim: the four counters, the
    undecided drafts, the minted spec names, and the episodic session's ledger path."""
    cycle = SessionActivity()

    cycle.fold_research(
        ResearchSummary(
            iterations=5,
            promotions=2,
            rejections=1,
            dead_ends=3,
            undecided=["draft_a"],
            minted_specs=["spec_x"],
            ledger_path="/state/sessions/s1.jsonl",
        )
    )

    assert cycle.research_iterations == 5
    assert cycle.research_promotions == 2
    assert cycle.research_rejections == 1
    assert cycle.research_dead_ends == 3
    assert cycle.research_undecided == ["draft_a"]
    assert cycle.minted_specs == ["spec_x"]
    assert cycle.research_ledgers == ["/state/sessions/s1.jsonl"]


def test_fold_research_accumulates_across_a_cycles_sessions():
    """A closed market runs research back to back, so a cycle folds several summaries: the
    counters add and the lists extend — never replace — and the cycle owns its own copies."""
    cycle = SessionActivity()
    first = ResearchSummary(
        iterations=2,
        promotions=1,
        undecided=["draft_a"],
        minted_specs=["spec_x"],
        ledger_path="/s1.jsonl",
    )

    cycle.fold_research(first)
    cycle.fold_research(
        ResearchSummary(
            iterations=3,
            rejections=2,
            dead_ends=1,
            undecided=["draft_b"],
            minted_specs=["spec_y"],
            ledger_path="/s2.jsonl",
        )
    )

    assert (cycle.research_iterations, cycle.research_promotions) == (5, 1)
    assert (cycle.research_rejections, cycle.research_dead_ends) == (2, 1)
    assert cycle.research_undecided == ["draft_a", "draft_b"]
    assert cycle.minted_specs == ["spec_x", "spec_y"]
    assert cycle.research_ledgers == ["/s1.jsonl", "/s2.jsonl"]
    # The cycle accumulated into its own lists; the folded summary is untouched.
    assert first.undecided == ["draft_a"] and first.minted_specs == ["spec_x"]


def test_fold_research_without_a_ledger_records_no_session():
    """The conversation loop and the legacy loop write no ledger, so nothing is appended —
    a ledgerless cycle's report renders exactly as it did before episodic sessions."""
    cycle = SessionActivity()

    cycle.fold_research(ResearchSummary(iterations=1))

    assert cycle.research_ledgers == []


def test_fold_trading_leaves_the_cycle_untouched_when_nothing_traded():
    """An empty outcome — account refusal, no new data, an empty champion board — contributes
    nothing: a skipped day reports zeros rather than a fictional flat session."""
    cycle = SessionActivity()

    cycle.fold_trading(TradingOutcome())

    assert (cycle.start_equity, cycle.end_equity) == (0.0, 0.0)
    assert cycle.positions == {} and cycle.trades == [] and cycle.events == []
    assert cycle.live_bars == {}


def test_fold_trading_states_why_it_did_not_trade_without_inventing_equity():
    """No session settled, but the outcome still says why: the reason is folded as a report
    event while equity and positions stay at their untouched zeros."""
    cycle = SessionActivity()

    cycle.fold_trading(TradingOutcome(events=["No new sessions to trade"]))

    assert cycle.events == ["No new sessions to trade"]
    assert (cycle.start_equity, cycle.end_equity) == (0.0, 0.0)
    assert cycle.positions == {}


def test_fold_trading_folds_a_settled_session():
    """One settled session hands over everything the close reads: equity either side,
    positions, trade rows, report events, and the bars the live feed built."""
    cycle = SessionActivity()
    bars = pd.DataFrame({"close": [190.0]})
    trade = Trade("AAPL", "buy", 10, 190.0, "champion signal")

    cycle.fold_trading(
        TradingOutcome(
            sessions=[TradingSummary(session=date(2026, 1, 6))],
            trades=[trade],
            events=["1 order refused by risk limits"],
            positions={"AAPL": 10.0},
            start_equity=100_000.0,
            end_equity=100_250.0,
            orders_submitted=1,
            live_bars={"AAPL": bars},
        )
    )

    assert (cycle.start_equity, cycle.end_equity) == (100_000.0, 100_250.0)
    assert cycle.positions == {"AAPL": 10.0}
    assert cycle.trades == [trade]
    assert cycle.events == ["1 order refused by risk limits"]
    assert list(cycle.live_bars) == ["AAPL"] and cycle.live_bars["AAPL"] is bars


def test_fold_trading_keeps_the_last_settled_sessions_equity():
    """A replay catch-up settles several sessions; the outcome already folded them, so the
    cycle carries that phase-level equity — the last session's — as handed over."""
    cycle = SessionActivity()

    cycle.fold_trading(
        TradingOutcome(
            sessions=[
                TradingSummary(session=date(2026, 1, 5)),
                TradingSummary(session=date(2026, 1, 6)),
            ],
            positions={"MSFT": -3.0},
            start_equity=100_000.0,
            end_equity=99_500.0,
        )
    )

    assert (cycle.start_equity, cycle.end_equity) == (100_000.0, 99_500.0)
    assert cycle.positions == {"MSFT": -3.0}


def test_a_folded_cycle_assembles_into_the_report(tmp_path):
    """The folds feed the report they exist for: a cycle that folded one research summary and
    one traded session assembles into the same ReportData the runtime produced by hand."""
    cycle = SessionActivity()
    cycle.fold_research(ResearchSummary(iterations=4, promotions=1, minted_specs=["spec_x"]))
    cycle.fold_trading(
        TradingOutcome(
            sessions=[TradingSummary(session=date(2026, 1, 6))],
            trades=[Trade("AAPL", "buy", 10, 190.0, "champion signal")],
            events=["1 order refused by risk limits"],
            positions={"AAPL": 10.0},
            start_equity=100_000.0,
            end_equity=100_250.0,
        )
    )

    data = assemble_report(
        as_of="2026-01-06",
        mode="paper",
        registry=ChampionRegistry(tmp_path / "champions.json", capacity=3),
        memory=InMemoryMemory(),
        state_dir=tmp_path / "state",
        session=cycle,
    )

    assert data.realized_pnl == pytest.approx(250.0)
    assert data.positions == {"AAPL": 10.0} and len(data.trades) == 1
    assert data.research["iterations"] == 4 and data.research["minted"] == ["spec_x"]
    assert data.events == ("1 order refused by risk limits",)
