"""The run's evidence: the six collectors, derived **once** per write (story #288, epic #284).

``evidence`` is the module that reads the run's own durable artifacts — its experiment journals,
its session ledgers, its champion board, its strategy tiers, its equity curve and the shared lake —
and it is the only module of the package allowed to name a heavy package. Everything asserted here
is external: the value :func:`derive_evidence` returns for a tree, the record a store writes, and
how many times a collector was entered.

The counting fake is deliberately patched onto :mod:`noctis.reporting.run_tree.evidence` — the
module that *defines* the six reads — because that is the one place every caller goes through. It
pins the epic's fourth problem statement closed: before the split, ``open_run`` ran the six
collectors and then ``RunStore.__init__`` ran them again before the first write, so a run with a
long journal and a big ledger read its own history (and the lake) twice for nothing.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from noctis.reporting import schema
from noctis.reporting.metrics import Benchmark, DailySession
from noctis.reporting.run_record import RunArtifacts, SpendEntry, StrategyArtifact
from noctis.reporting.run_tree import (
    RUN_RECORD_NAME,
    Evidence,
    derive_evidence,
    finish_run,
    open_run,
)
from noctis.reporting.run_tree import evidence as evidence_module
from noctis.reporting.run_tree.store import with_evidence

from ._run_tree_helpers import ENGINE, FakeClock

# The six reads, in the order the record derives them — the pass a write is allowed exactly one of.
COLLECTORS = (
    "read_trials",
    "read_spend",
    "read_champions",
    "read_strategies",
    "read_sessions",
    "read_benchmark",
)


def _open(runs_dir: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _artifacts(**changes: object) -> RunArtifacts:
    """Artifacts carrying nothing but what a test hands in — every derived field at its default."""
    fields: dict[str, object] = {
        "run_id": "20260727T142233Z-a1b2c3",
        "created_utc": None,
        "last_active_utc": None,
        "engine": ENGINE,
    }
    return RunArtifacts(**{**fields, **changes})  # type: ignore[arg-type]


def _journal_trial(run_dir: Path, strategy: str, **params) -> None:
    """Journal one trial into the run's own experiment journal — the exhaustion gate's ground
    truth, and the only place the record's trial count is ever read from."""
    from noctis.research.journal import ExperimentJournal
    from tests.test_champions import make_scorecard

    ExperimentJournal(run_dir / "state").record_trial(
        strategy,
        source="sweep",
        symbols=["AAPL"],
        params=params,
        window={},
        card=make_scorecard(strategy, test_metric=1.2, train_metric=1.4),
    )


@pytest.fixture
def passes(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count every entry into the six collectors, without changing what any of them returns."""
    counted: Counter[str] = Counter()

    def counting(name: str, real: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            counted[name] += 1
            return real(*args, **kwargs)

        return wrapper

    for name in COLLECTORS:
        monkeypatch.setattr(evidence_module, name, counting(name, getattr(evidence_module, name)))
    return counted


# ── one pass per write ─────────────────────────────────────────────────────────────────────


def test_opening_a_run_derives_its_evidence_exactly_once(tmp_path, passes):
    """The epic's fourth problem, closed: an open used to run the six reads, and then the write
    that ended it ran all six again and overwrote every one of them."""
    store = _open(tmp_path / "runs", FakeClock())

    assert dict(passes) == dict.fromkeys(COLLECTORS, 1)
    assert store.record()["run"]["cumulative_trials"] is None


def test_each_checkpoint_derives_the_evidence_once_more(tmp_path, passes):
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)

    store.checkpoint(counters={"cycles": 1})

    assert dict(passes) == dict.fromkeys(COLLECTORS, 2)


def test_closing_a_segment_derives_the_evidence_once_more(tmp_path, passes):
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)
    clock.advance(3600)

    store.close(reason="time_limit")

    assert dict(passes) == dict.fromkeys(COLLECTORS, 2)


def test_sealing_a_run_derives_its_evidence_once(tmp_path, passes):
    """``finish_run`` writes the record without a store, so it needs the evidence itself."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(3600)
    store.close(reason="time_limit")
    passes.clear()

    finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    assert dict(passes) == dict.fromkeys(COLLECTORS, 1)


# ── derive_evidence: what an empty tree knows about itself ──────────────────────────────────


def test_an_empty_run_tree_derives_the_empty_evidence(tmp_path):
    """Every field at the value ``RunArtifacts`` defaults to — except the benchmark, which says
    in words that a run which traded nothing has nothing to be compared against."""
    run_dir = tmp_path / "runs" / "20260727T142233Z-a1b2c3"
    run_dir.mkdir(parents=True)

    derived = derive_evidence(run_dir, None)

    assert derived == Evidence(
        trials=None,
        spend=None,
        pricing_table_version=None,
        champions=None,
        strategies=(),
        sessions=(),
        benchmark=Benchmark(symbols=(), points=(), note="this run has traded nothing to benchmark"),
    )


def test_the_empty_evidence_matches_what_the_record_defaults_to(tmp_path):
    """A field whose empty value here differs from the builder's default would change a record."""
    run_dir = tmp_path / "runs" / "20260727T142233Z-a1b2c3"
    run_dir.mkdir(parents=True)
    defaults = _artifacts(run_id=run_dir.name)

    derived = derive_evidence(run_dir, None)

    carried = {name: value for name, value in derived.changes().items() if name != "benchmark"}
    assert carried == {name: getattr(defaults, name) for name in carried}


# ── with_evidence: every derived field lands on the artifacts ───────────────────────────────

# One value per field of :class:`Evidence`, every one of them different from what
# ``RunArtifacts`` defaults to, so a field the copy forgets fails instead of matching by accident.
EPISODE = SpendEntry(at="2026-07-27T15:00:00.000Z", stage="decide", model="acme/oracle", tokens=12)
SAMPLES: dict[str, object] = {
    "trials": 7,
    "spend": (EPISODE,),
    "pricing_table_version": "2026-07.1+custom.deadbeef",
    "champions": 3,
    "strategies": (StrategyArtifact(name="momo_1", outcome="promoted", tier="champions"),),
    "sessions": (DailySession(date="2026-07-28", equity=100_000.0),),
    "benchmark": Benchmark(symbols=("NVDA",), points=(("2026-07-28", 1.0),)),
}


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(Evidence)])
def test_with_evidence_carries_every_derived_field_onto_the_artifacts(field, sample_evidence):
    updated = with_evidence(_artifacts(), sample_evidence)

    assert getattr(updated, field) == SAMPLES[field]


@pytest.fixture
def sample_evidence() -> Evidence:
    return Evidence(**SAMPLES)  # type: ignore[arg-type]


def test_with_evidence_changes_nothing_else_about_the_artifacts(sample_evidence):
    """A copy, not a rebuild: everything the artifacts already carried is still there."""
    artifacts = _artifacts(
        created_utc="2026-07-27T14:22:33.418Z",
        last_active_utc="2026-07-27T15:22:33.418Z",
        label="nightly-momo",
        state_pruned=True,
        inputs={"settings": {"resolved": {}}},
    )

    updated = with_evidence(artifacts, sample_evidence)

    assert updated.label == "nightly-momo"
    assert updated.state_pruned is True
    assert updated.inputs == {"settings": {"resolved": {}}}
    assert updated.created_utc == "2026-07-27T14:22:33.418Z"


def test_the_evidence_reports_its_changes_without_recursing_into_its_values(sample_evidence):
    """``changes()`` is a shallow ``{field: value}`` — ``dataclasses.asdict`` would turn every
    ``SpendEntry``, ``StrategyArtifact`` and ``DailySession`` into a dict the builder cannot
    read."""
    changes = sample_evidence.changes()

    assert set(changes) == {f.name for f in dataclasses.fields(Evidence)}
    assert changes["benchmark"] is sample_evidence.benchmark


# ── the evidence a real run leaves, read back off the record it wrote ───────────────────────


def test_the_derived_evidence_is_what_the_record_publishes(tmp_path):
    """The two verbs meet on disk: what ``derive_evidence`` reads is what the store writes."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    _journal_trial(store.run_dir, "alpha", lookback=10)
    store.checkpoint()

    derived = derive_evidence(store.run_dir, None)

    assert derived.trials == 1
    assert _record(store.run_dir)["run"]["cumulative_trials"] == 1
    assert schema.validate(_record(store.run_dir)) == []


# ── the trials a run journals, counted off its own journals (story #137) ───────────────────


def test_the_record_counts_the_trials_the_run_journaled(tmp_path):
    """The multiple-testing count comes from the run's own journals — the same lines the
    exhaustion gate counts — and is re-read at every write rather than incremented."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    assert _record(store.run_dir)["run"]["cumulative_trials"] is None

    _journal_trial(store.run_dir, "alpha", lookback=10)
    _journal_trial(store.run_dir, "alpha", lookback=20)
    _journal_trial(store.run_dir, "beta", lookback=5)
    store.checkpoint()

    assert _record(store.run_dir)["run"]["cumulative_trials"] == 3
    clock.advance(3600)
    store.close(reason="stopped")
    assert _record(store.run_dir)["run"]["cumulative_trials"] == 3


def test_the_trial_count_is_re_read_from_the_journals_never_carried_across_a_restart(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    _journal_trial(first.run_dir, "alpha", lookback=10)
    clock.advance(3600)
    first.close(reason="time_limit")

    second = _open(runs, clock, run_id=first.run_id, resume=True, command="research")
    _journal_trial(second.run_dir, "alpha", lookback=20)
    clock.advance(1800)
    second.close(reason="agent_done")

    # Nothing was handed forward: the resumed process read both segments' trials off disk.
    assert _record(second.run_dir)["run"]["cumulative_trials"] == 2


# ── spend, read off the run's own ledgers and board (story #140) ───────────────────────────


def _journal_episode(run_dir: Path, session: str, *, stage: str, model: str, **usage) -> None:
    """Journal one model judgment into the run's own session ledger — the only place the record's
    spend is ever read from."""
    from noctis.research.ledger import SessionLedger

    split = {
        "input_tokens": usage.get("inp", 0),
        "output_tokens": usage.get("out", 0),
        "cache_creation_input_tokens": usage.get("write", 0),
        "cache_read_input_tokens": usage.get("read", 0),
    }
    SessionLedger(run_dir / "state", session).record_episode(
        stage=stage,
        model=model,
        outcome="ok",
        tokens=sum(split.values()),
        usage=split,
    )


def _crown(run_dir: Path, *families: str) -> None:
    """Crown champions on the run's own board — the denominator of the per-champion numbers."""
    from noctis.champions.registry import ChampionEntry, ChampionRegistry
    from tests.test_champions import make_scorecard

    registry = ChampionRegistry(run_dir / "state" / "champions.json", capacity=len(families))
    registry.champions = [
        ChampionEntry(
            family=name,
            params={},
            scorecard=make_scorecard(name, 1.2, 1.4),
            crowned_at="2026-07-27T15:00:00+00:00",
            rationale="free slot",
        )
        for name in families
    ]
    registry.save()


def test_the_record_derives_its_spend_from_the_runs_own_session_ledgers(tmp_path):
    """Token usage was computed and then only logged; now it is read back off the ledgers that
    journaled it, so cost per champion is recoverable from the artifact alone (story #140)."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    assert _record(store.run_dir)["spend"] is None  # nothing journaled yet ⇒ null, not zeros

    _journal_episode(
        store.run_dir, "s1", stage="formulate", model="anthropic/claude-opus-4-8", inp=1000, out=200
    )
    _journal_episode(
        store.run_dir, "s1", stage="decide", model="anthropic/claude-opus-4-8", inp=500, read=4000
    )
    _journal_episode(
        store.run_dir, "s2", stage="author", model="ollama/qwen3-coder-30b", inp=9000, out=3000
    )
    _crown(store.run_dir, "momo_1", "drift_2")
    _journal_trial(store.run_dir, "momo_1", lookback=10)
    clock.advance(3600)
    store.close(reason="time_limit", phase_seconds={"RESEARCH": 3600.0})

    spend = _record(store.run_dir)["spend"]
    assert spend["tokens"] == {
        "input_tokens": 10500,
        "output_tokens": 3200,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 4000,
        "total_tokens": 17700,
    }
    assert set(spend["by_model"]) == {"anthropic/claude-opus-4-8", "ollama/qwen3-coder-30b"}
    assert set(spend["by_stage"]) == {"formulate", "decide", "author"}
    assert spend["by_model"]["ollama/qwen3-coder-30b"]["usd_estimate"] == 0.0  # a stated zero
    assert spend["llm_usd_estimate"] > 0
    # The efficiency numbers, over the run's own champions, trials and research hours.
    assert spend["efficiency"]["usd_per_champion_estimate"] == round(
        spend["llm_usd_estimate"] / 2, 6
    )
    assert spend["efficiency"]["trials_per_hour"] == 1.0
    assert schema.validate(_record(store.run_dir)) == []


def test_the_record_names_the_price_table_that_produced_its_estimate(tmp_path):
    from noctis.research.pricing import PRICING_TABLE_VERSION

    runs = tmp_path / "runs"
    store = _open(runs, FakeClock())
    _journal_episode(store.run_dir, "s1", stage="decide", model="openai/gpt-5.4", inp=10, out=2)
    store.checkpoint()

    assert _record(store.run_dir)["spend"]["pricing_table_version"] == PRICING_TABLE_VERSION


def test_an_operator_price_override_prices_the_run_and_says_the_table_is_no_longer_the_shipped_one(
    tmp_path,
):
    """The override travels with the run's frozen config, and it cannot borrow the shipped
    version label — a reader must always be able to tell whose prices these are."""
    from noctis.research.pricing import PRICING_TABLE_VERSION

    inputs = {
        "settings": {
            "resolved": {
                "research": {
                    "pricing": {
                        "acme/oracle": {
                            "input_usd_per_mtok": 1_000_000.0,
                            "output_usd_per_mtok": 0.0,
                            "cache_write_usd_per_mtok": 0.0,
                            "cache_read_usd_per_mtok": 0.0,
                        }
                    }
                }
            }
        }
    }
    runs = tmp_path / "runs"
    store = _open(runs, FakeClock(), inputs=inputs)
    _journal_episode(store.run_dir, "s1", stage="decide", model="acme/oracle-1", inp=1_000_000)
    store.checkpoint()

    spend = _record(store.run_dir)["spend"]
    assert spend["llm_usd_estimate"] == 1_000_000.0
    assert spend["pricing_table_version"].startswith(f"{PRICING_TABLE_VERSION}+custom.")


def test_a_model_the_price_table_never_heard_of_costs_null_not_zero(tmp_path):
    runs = tmp_path / "runs"
    store = _open(runs, FakeClock())
    _journal_episode(store.run_dir, "s1", stage="decide", model="acme/oracle-1", inp=10, out=2)
    _crown(store.run_dir, "momo_1")
    store.checkpoint()

    spend = _record(store.run_dir)["spend"]
    assert spend["tokens"]["total_tokens"] == 12  # the tokens are known
    assert spend["by_model"]["acme/oracle-1"]["usd_estimate"] is None
    assert spend["llm_usd_estimate"] is None
    assert spend["efficiency"]["usd_per_champion_estimate"] is None


def test_spend_is_re_derived_from_the_ledgers_never_carried_across_a_restart(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    _journal_episode(first.run_dir, "s1", stage="decide", model="openai/gpt-5.4", inp=100)
    clock.advance(3600)
    first.close(reason="time_limit")

    second = _open(runs, clock, run_id=first.run_id, resume=True, command="research")
    _journal_episode(second.run_dir, "s2", stage="decide", model="openai/gpt-5.4", inp=300)
    clock.advance(1800)
    second.close(reason="agent_done")

    # Nothing was handed forward: the resumed process read both nights' ledgers off disk.
    assert _record(second.run_dir)["spend"]["tokens"]["input_tokens"] == 400


def test_an_unreadable_ledger_costs_the_record_its_spend_not_the_run(tmp_path):
    """Missing evidence is not a reason to fail a write — the record says it does not know."""
    runs = tmp_path / "runs"
    store = _open(runs, FakeClock())
    sessions = store.run_dir / "state" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "s1.jsonl").write_text("{not json at all\n")
    store.checkpoint()

    record = _record(store.run_dir)
    assert record["spend"]["tokens"]["total_tokens"] == 0  # a malformed line is skipped, not fatal
    assert schema.validate(record) == []
