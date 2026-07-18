"""RESEARCH loop: budget/stop control, promotion wiring, pruning, determinism, findings."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.champions import ChampionRegistry, PromotionRules
from noctis.engine import run_research
from noctis.memory import InMemoryMemory
from noctis.strategies import Candidate, CandidateProposer, signature

RULES = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)


def _sc(family: str, test: float, train: float, stage: str = "validated", **params) -> Scorecard:
    def m(x):
        return Metrics(x, x, 0.0, 0.0, 0.0, 0.0, 0.0)

    symbols = (
        {}
        if stage != "validated"
        else {"FIT": SymbolScore(splits=[SplitScore(0, m(train), m(test))])}
    )
    return Scorecard(family=family, params=params, stage=stage, symbols=symbols)


class FakeClock:
    """Deterministic clock advancing a fixed step each call."""

    def __init__(self, step_seconds: float = 30.0):
        self.base = datetime(2026, 1, 1, tzinfo=UTC)
        self.elapsed = 0.0
        self.step = step_seconds

    def __call__(self) -> datetime:
        t = self.base + timedelta(seconds=self.elapsed)
        self.elapsed += self.step
        return t


class StopAfter:
    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def is_set(self) -> bool:
        fire = self.calls >= self.n
        self.calls += 1
        return fire


# --- 1. budget stop, promotion of the good one -------------------------------------------


def test_loop_runs_promotes_and_stops_on_budget(tmp_path):
    counter = itertools.count()

    def evaluate_fn(cand: Candidate) -> Scorecard:
        i = next(counter)
        # First candidate is excellent; the rest are below the bar.
        return _sc(cand.family, test=2.0 if i == 0 else -0.5, train=2.1 if i == 0 else -0.4)

    registry = ChampionRegistry(tmp_path / "champs.json", capacity=3)
    memory = InMemoryMemory()
    summary = run_research(
        proposer=CandidateProposer(seed=1),
        evaluate_fn=evaluate_fn,
        registry=registry,
        rules=RULES,
        memory=memory,
        budget_minutes=3.0,
        now=FakeClock(step_seconds=30.0),
    )
    assert summary.stopped_reason == "time_budget"
    assert summary.iterations == 5  # 30s steps until 180s budget
    assert summary.promotions == 1
    assert summary.rejections == 4
    assert len(registry.list()) == 1


# --- 2. stop event mid-loop --------------------------------------------------------------


def test_stop_event_exits_cleanly(tmp_path):
    def evaluate_fn(cand: Candidate) -> Scorecard:
        return _sc(cand.family, test=1.0, train=1.0)

    registry = ChampionRegistry(tmp_path / "champs.json", capacity=3)
    summary = run_research(
        proposer=CandidateProposer(seed=2),
        evaluate_fn=evaluate_fn,
        registry=registry,
        rules=RULES,
        memory=InMemoryMemory(),
        budget_minutes=10_000.0,  # budget won't be the reason
        now=FakeClock(step_seconds=1.0),
        stop_event=StopAfter(2),
    )
    assert summary.stopped_reason == "stop_event"
    assert summary.iterations == 2
    # Registry persisted and reloadable → consistent after interruption.
    reloaded = ChampionRegistry(tmp_path / "champs.json", capacity=3)
    assert len(reloaded.list()) == len(registry.list())


# --- 3. rejected-ideas pruning -----------------------------------------------------------


def test_proposer_prunes_rejected_ideas():
    memory = InMemoryMemory()
    memory.record_rejected("sma_crossover", {"fast": 5, "slow": 20})
    proposer = CandidateProposer(seed=7, memory=memory)
    sma_space = proposer.families.param_space("sma_crossover")
    rejected_sig = signature("sma_crossover", sma_space, {"fast": 5, "slow": 20})
    for _ in range(60):
        cand = proposer.propose()
        sig = signature(cand.family, proposer.families.param_space(cand.family), cand.params)
        assert sig != rejected_sig  # never re-proposes the known dead end


# --- 4. determinism ----------------------------------------------------------------------


def test_proposer_is_deterministic_for_fixed_seed():
    def seq(seed):
        p = CandidateProposer(seed=seed)
        out = []
        for _ in range(12):
            c = p.propose()
            out.append((c.family, tuple(sorted(c.params.items()))))
        return out

    assert seq(123) == seq(123)
    assert seq(123) != seq(999)  # different seeds diverge


# --- 5. findings appended ----------------------------------------------------------------


def test_findings_recorded_for_promotion_and_dead_end(tmp_path):
    scripted = iter(
        [
            _sc("sma_crossover", test=2.0, train=2.1),  # promoted
            _sc("rsi_meanrev", test=0.0, train=0.0, stage="prefilter_rejected"),  # dead end
        ]
    )

    def evaluate_fn(_cand: Candidate) -> Scorecard:
        return next(scripted)

    memory = InMemoryMemory()
    summary = run_research(
        proposer=CandidateProposer(seed=3),
        evaluate_fn=evaluate_fn,
        registry=ChampionRegistry(tmp_path / "c.json", capacity=3),
        rules=RULES,
        memory=memory,
        budget_minutes=1000.0,
        now=FakeClock(step_seconds=1.0),
        max_iterations=2,
    )
    assert summary.iterations == 2
    assert summary.promotions == 1
    assert summary.dead_ends == 1
    findings = memory.findings()
    assert any("PROMOTED" in f for f in findings)
    assert any("DEAD END" in f for f in findings)
    assert memory.rejected_ideas()  # the dead end was remembered


def test_max_iterations_caps_loop(tmp_path):
    def evaluate_fn(cand: Candidate) -> Scorecard:
        return _sc(cand.family, test=0.1, train=0.1)

    summary = run_research(
        proposer=CandidateProposer(seed=4),
        evaluate_fn=evaluate_fn,
        registry=ChampionRegistry(tmp_path / "c.json", capacity=3),
        rules=RULES,
        memory=InMemoryMemory(),
        budget_minutes=1000.0,
        now=FakeClock(step_seconds=1.0),
        max_iterations=3,
    )
    assert summary.iterations == 3
    assert summary.stopped_reason == "max_iterations"


# --- 7. a hung evaluation is absorbed, never propagated ----------------------------------


def test_hung_evaluation_is_a_dead_end_not_a_crash(tmp_path):
    """Nothing above run_research catches research exceptions — an EvaluationTimeout escaping
    here would kill the whole `noctis run`. The loop must absorb it as a dead end (candidate
    rejected, finding recorded) and keep iterating within its ordinary budgets."""
    from noctis.backtest.pool import EvaluationTimeout

    counter = itertools.count()

    def evaluate_fn(cand: Candidate) -> Scorecard:
        if next(counter) == 1:  # the second candidate hangs; the guard bounds it
            raise EvaluationTimeout("evaluation exceeded 1800s wall-clock")
        return _sc(cand.family, test=-0.5, train=-0.4)

    memory = InMemoryMemory()
    summary = run_research(
        proposer=CandidateProposer(seed=7),
        evaluate_fn=evaluate_fn,
        registry=ChampionRegistry(tmp_path / "champs.json", capacity=3),
        rules=RULES,
        memory=memory,
        budget_minutes=2.0,
        now=FakeClock(step_seconds=30.0),
    )
    assert summary.stopped_reason == "time_budget"  # the loop survived the hung candidate
    assert summary.dead_ends >= 1
    assert any("evaluation hung" in f for f in memory.findings())
