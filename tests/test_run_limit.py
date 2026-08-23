"""The run-level compute cap and the explicit finish (story #136, epic #126).

Two ways a run reaches ``completed`` — the one terminal status — and both exist for the same
reason: **a published result may not silently gain segments**.

* ``--run-limit-hours`` is frozen at run creation and bounds the *run*, not the process. Once the
  runtime the run has accumulated across every segment breaches it, the loop stops between phases
  through the shutdown path ``time_limit_hours`` already uses, and the record reads ``completed``.
  That is what makes two runs comparable **on equal compute**: a mandate given 100 hours and one
  given 30 are not the same experiment.
* ``--finish`` seals a run deliberately, running no segment at all.

Everything asserted here is external — what the record on disk says, what the next resume does,
what the CLI prints. The clock is always injected; no test reads a wall clock.
"""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.reporting.run_record import (
    EngineIdentity,
    RunArtifacts,
    SegmentArtifact,
    build,
    resume_refusal,
)

runner = CliRunner()

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
HOUR = 3600.0

# A cycle cap the tests below must never reach: every one of them asserts the loop stopped for a
# *limit*, so this is only a net that turns "the limit never fired" into a failed assertion instead
# of a hung suite.
_NEVER_REACHED = 20

ENGINE = EngineIdentity(
    engine_version=1,
    fingerprint={"gates": "f63d47b7b9604ab1", "backtest": "3ba3e0bf1c97134f"},
    comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe",
    noctis_version="0.1.0",
)


# ── in-memory fixtures ─────────────────────────────────────────────────────────────────────


def _stamp(offset_s: float) -> str:
    moment = START + timedelta(seconds=offset_s)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


def _segment(index: int, *, start_s: float, seconds: float | None, **overrides) -> SegmentArtifact:
    """One closed segment starting ``start_s`` after the run, or an open one (``seconds=None``)."""
    base = dict(
        index=index,
        started_utc=_stamp(start_s),
        stopped_utc=None if seconds is None else _stamp(start_s + seconds),
        stopped_reason=None if seconds is None else "time_limit",
        status="running" if seconds is None else "stopped",
        argv=("run", "-v"),
        command="run",
        resumed=index > 0,
        counters={"cycles": 1},
        engine=ENGINE,
    )
    base.update(overrides)
    return SegmentArtifact(**base)  # type: ignore[arg-type]


def _frozen(**resolved) -> dict:
    """The shape ``config.rehydrate.freeze_inputs`` writes, reduced to what the cap needs."""
    return {"settings": {"resolved": resolved}}


def _artifacts(*segments: SegmentArtifact, inputs: dict | None = None) -> RunArtifacts:
    return RunArtifacts(
        run_id="20260727T142233Z-a1b2c3",
        created_utc=_stamp(0),
        last_active_utc=_stamp(0),
        engine=ENGINE,
        segments=segments,
        complete=True,
        inputs=inputs,
    )


# ── the cap is frozen configuration, surfaced on the record ────────────────────────────────


def test_the_cap_belongs_to_the_frozen_tier_and_the_process_limit_stays_live():
    """The two ceilings sit in different tiers *because* they bound different things: the run's
    compute budget defines the experiment (frozen at creation), while how long tonight lasts is the
    operator's call every night (live). Asserted here rather than assumed, since the frozen tier is
    a complement — a new knob freezes by default, and this is the story that relies on it."""
    from noctis.config.rehydrate import FROZEN, LIVE, classify

    assert classify("run_limit_hours") == "frozen"
    assert "run_limit_hours" in FROZEN
    assert classify("time_limit_hours") == "live"
    assert "time_limit_hours" in LIVE


def test_the_record_reports_the_cap_the_runs_frozen_configuration_carries():
    record = build(
        _artifacts(_segment(0, start_s=0, seconds=HOUR), inputs=_frozen(run_limit_hours=100))
    )

    assert record["run"]["run_limit_hours"] == 100.0


def test_a_run_with_no_cap_reports_an_explicit_null_rather_than_omitting_the_key():
    uncapped = build(_artifacts(_segment(0, start_s=0, seconds=HOUR), inputs=_frozen()))
    unfrozen = build(_artifacts(_segment(0, start_s=0, seconds=HOUR)))

    assert uncapped["run"]["run_limit_hours"] is None
    assert unfrozen["run"]["run_limit_hours"] is None
    assert "run_limit_hours" in uncapped["run"] and "run_limit_hours" in unfrozen["run"]


def test_the_cap_the_record_reports_is_the_one_freeze_inputs_wrote(tmp_path):
    """The record does not keep its own copy of the cap: it reads the run's frozen settings, so
    the number an operator sees is by construction the number the engine enforces."""
    from noctis.config import load_settings
    from noctis.config.rehydrate import freeze_inputs

    path = tmp_path / "config.yaml"
    path.write_text("mode: paper\nrun_limit_hours: 12.5\n")
    inputs = freeze_inputs(load_settings(config_path=path), frozen_at=_stamp(0))

    record = build(_artifacts(_segment(0, start_s=0, seconds=HOUR), inputs=inputs))

    assert record["run"]["run_limit_hours"] == 12.5


# ── cumulative research / trading seconds are DERIVED from segments[] ──────────────────────


def test_research_and_trading_seconds_are_summed_from_the_segments_own_phase_timings():
    record = build(
        _artifacts(
            _segment(
                0,
                start_s=0,
                seconds=HOUR,
                phase_seconds={"RESEARCH": 2400.0, "TRADING": 900.0, "CLOSE": 12.0},
            ),
            _segment(
                1,
                start_s=8 * HOUR,
                seconds=2 * HOUR,
                phase_seconds={"RESEARCH": 6000.0, "TRADING": 0.0, "CLOSE": 8.0},
            ),
        )
    )

    assert record["run"]["cumulative_research_s"] == 8400.0
    assert record["run"]["cumulative_trading_s"] == 900.0
    assert record["run"]["cumulative_runtime_s"] == 3 * HOUR
    assert record["segments"][0]["phase_seconds"] == {
        "RESEARCH": 2400.0,
        "TRADING": 900.0,
        "CLOSE": 12.0,
    }


def test_three_short_segments_carry_the_same_phase_totals_as_one_long_one():
    """Derived, never incremented (epic D4): splitting a night into three cannot move a total."""
    one = build(
        _artifacts(
            _segment(
                0,
                start_s=0,
                seconds=3 * HOUR,
                phase_seconds={"RESEARCH": 9000.0, "TRADING": 1800.0},
            )
        )
    )
    three = build(
        _artifacts(
            *(
                _segment(
                    i,
                    start_s=i * HOUR,
                    seconds=HOUR,
                    phase_seconds={"RESEARCH": 3000.0, "TRADING": 600.0},
                )
                for i in range(3)
            )
        )
    )

    for key in ("cumulative_runtime_s", "cumulative_research_s", "cumulative_trading_s"):
        assert one["run"][key] == three["run"][key]
    assert one["run"]["cumulative_research_s"] == 9000.0


def test_a_run_whose_segments_recorded_no_phase_timings_reports_null_not_zero():
    """A record written before phase timings existed knows nothing about them; ``0.0`` would be a
    claim it never made. A segment that *did* record them and never traded reports an honest 0."""
    silent = build(_artifacts(_segment(0, start_s=0, seconds=HOUR)))
    researched_only = build(
        _artifacts(_segment(0, start_s=0, seconds=HOUR, phase_seconds={"RESEARCH": 3000.0}))
    )

    assert silent["run"]["cumulative_research_s"] is None
    assert silent["run"]["cumulative_trading_s"] is None
    assert silent["segments"][0]["phase_seconds"] is None
    assert researched_only["run"]["cumulative_research_s"] == 3000.0
    assert researched_only["run"]["cumulative_trading_s"] == 0.0


# ── breaching the cap seals the run ────────────────────────────────────────────────────────


def test_a_run_whose_cumulative_runtime_breaches_the_cap_reads_completed():
    record = build(
        _artifacts(
            _segment(0, start_s=0, seconds=2 * HOUR),
            _segment(1, start_s=8 * HOUR, seconds=1.5 * HOUR),
            inputs=_frozen(run_limit_hours=3),
        )
    )

    assert record["run"]["status"] == "completed"
    # …stamped at the moment it crossed, which is the close of the segment that took it over.
    assert record["run"]["completed_utc"] == _stamp(9.5 * HOUR)
    assert resume_refusal(record) is not None


def test_a_segment_ending_below_the_cap_leaves_the_run_stopped_and_resumable():
    record = build(
        _artifacts(
            _segment(0, start_s=0, seconds=2 * HOUR),
            inputs=_frozen(run_limit_hours=3),
        )
    )

    assert record["run"]["status"] == "stopped"
    assert record["run"]["completed_utc"] is None
    assert resume_refusal(record) is None


def test_an_open_segment_does_not_count_toward_the_cap_until_it_closes():
    """The cap is derived from closed segments, so a run is sealed at a phase boundary — never
    mid-write, where the duration is not yet an honest number."""
    open_segment = build(
        _artifacts(
            _segment(0, start_s=0, seconds=4 * HOUR),
            _segment(1, start_s=8 * HOUR, seconds=None),
            inputs=_frozen(run_limit_hours=3),
        )
    )

    assert open_segment["run"]["status"] == "completed"  # segment 0 alone already crossed
    assert open_segment["run"]["completed_utc"] == _stamp(4 * HOUR)


def test_an_uncapped_run_is_never_completed_by_runtime_alone():
    record = build(
        _artifacts(
            _segment(0, start_s=0, seconds=500 * HOUR),
            inputs=_frozen(run_limit_hours=None),
        )
    )

    assert record["run"]["status"] == "stopped"
    assert resume_refusal(record) is None


def test_a_deliberate_seal_still_wins_over_the_derivation():
    """``--finish`` on a run far below its cap is still terminal — the stamp it wrote stands."""
    sealed = build(
        RunArtifacts(
            run_id="20260727T142233Z-a1b2c3",
            created_utc=_stamp(0),
            last_active_utc=_stamp(0),
            engine=ENGINE,
            segments=(_segment(0, start_s=0, seconds=HOUR),),
            complete=True,
            completed_utc=_stamp(2 * HOUR),
            inputs=_frozen(run_limit_hours=100),
        )
    )

    assert sealed["run"]["status"] == "completed"
    assert sealed["run"]["completed_utc"] == _stamp(2 * HOUR)


def test_the_schema_accepts_the_capped_records_new_keys(tmp_path):
    from noctis.config import load_settings
    from noctis.config.rehydrate import freeze_inputs
    from noctis.reporting import schema

    path = tmp_path / "config.yaml"
    path.write_text("mode: paper\nrun_limit_hours: 1\n")
    inputs = freeze_inputs(load_settings(config_path=path), frozen_at=_stamp(0))

    record = build(
        _artifacts(
            _segment(0, start_s=0, seconds=HOUR, phase_seconds={"RESEARCH": 60.0}),
            inputs=inputs,
        )
    )

    assert schema.validate(record) == []
    assert record["run"]["status"] == "completed"


# ── the store: the totals survive a write, a kill and the next open ────────────────────────


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _open(runs_dir: Path, clock: FakeClock, **kwargs):
    from noctis.reporting.run_tree import open_run

    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _record(run_dir: Path) -> dict:
    from noctis.reporting.run_tree import RUN_RECORD_NAME

    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _capped_inputs(tmp_path: Path, hours: float | None) -> dict:
    from noctis.config import load_settings
    from noctis.config.rehydrate import freeze_inputs

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    limit = "null" if hours is None else str(hours)
    path.write_text(f"mode: paper\nrun_limit_hours: {limit}\n")
    return freeze_inputs(load_settings(config_path=path), frozen_at=_stamp(0))


def test_phase_timings_are_written_per_segment_and_summed_across_a_resume(tmp_path):
    """The run-level totals are read back off the segments on **disk**, so a resumed process adds
    its own night to a total it never held in memory."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    clock.advance(HOUR)
    first.close(reason="time_limit", phase_seconds={"RESEARCH": 3000.0, "TRADING": 500.0})

    clock.advance(8 * HOUR)
    second = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(HOUR)
    second.close(reason="stop_requested", phase_seconds={"RESEARCH": 2000.0, "TRADING": 100.0})

    record = _record(first.run_dir)
    assert [s["phase_seconds"] for s in record["segments"]] == [
        {"RESEARCH": 3000.0, "TRADING": 500.0},
        {"RESEARCH": 2000.0, "TRADING": 100.0},
    ]
    assert record["run"]["cumulative_research_s"] == 5000.0
    assert record["run"]["cumulative_trading_s"] == 600.0


def test_a_store_reports_the_runtime_the_earlier_segments_already_accumulated(tmp_path):
    """What a resumed process needs to enforce a run-level cap: the run's runtime *before* this
    segment. Derived from the record's closed segments, so the open one cannot inflate it."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    clock.advance(2 * HOUR)
    first.close(reason="time_limit")

    clock.advance(HOUR)
    second = _open(runs, clock, run_id=first.run_id, resume=True)

    assert first.prior_runtime_s == 0.0
    assert second.prior_runtime_s == 2 * HOUR


def test_a_run_that_breaches_its_cap_while_running_is_completed_at_its_segment_close(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    inputs = _capped_inputs(tmp_path / "cfg", 2)
    first = _open(runs, clock, inputs=inputs)
    clock.advance(1.5 * HOUR)
    first.close(reason="time_limit")
    assert _record(first.run_dir)["run"]["status"] == "stopped"

    second = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(HOUR)
    second.close(reason="run_limit")

    record = _record(first.run_dir)
    assert record["run"]["status"] == "completed"
    assert record["run"]["run_limit_hours"] == 2.0
    assert record["segments"][1]["stopped_reason"] == "run_limit"


def test_a_capped_out_run_refuses_the_next_resume_and_says_why(tmp_path):
    from noctis.reporting.run_tree import RunCompletedError

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_capped_inputs(tmp_path / "cfg", 1))
    clock.advance(2 * HOUR)
    store.close(reason="run_limit")

    try:
        _open(runs, clock, run_id=store.run_id, resume=True)
    except RunCompletedError as exc:
        refusal = str(exc)
    else:  # pragma: no cover - the refusal is the behaviour under test
        raise AssertionError("a capped-out run must refuse the next resume")

    assert store.run_id in refusal
    assert "1h run limit" in refusal
    assert "2h" in refusal


def test_the_index_lists_the_cap_beside_the_runtime_it_bounds(tmp_path):
    from noctis.reporting.run_tree import index_entry

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_capped_inputs(tmp_path / "cfg", 100))
    clock.advance(HOUR)
    store.close(reason="time_limit")

    entry = index_entry(store.run_dir)

    assert entry["run_limit_hours"] == 100.0
    assert entry["cumulative_runtime_s"] == HOUR
    assert index_entry(runs / "20260101T000000Z-missing")["run_limit_hours"] is None


# ── --finish: sealing a run without running a segment ──────────────────────────────────────


def test_finishing_a_run_seals_it_without_opening_a_segment(tmp_path):
    from noctis.reporting.run_tree import finish_run

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    before = _record(store.run_dir)

    clock.advance(HOUR)
    outcome = finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    record = _record(store.run_dir)
    assert outcome.sealed is True and outcome.run_id == store.run_id
    assert record["run"]["status"] == "completed"
    assert record["run"]["completed_utc"] == _stamp(2 * HOUR)
    # …and it ran nothing: same segments, same totals, no new lock.
    assert record["segments"] == before["segments"]
    assert record["run"]["cumulative_runtime_s"] == before["run"]["cumulative_runtime_s"]
    assert not (store.run_dir / "run.lock").exists()


def test_finishing_an_already_completed_run_is_a_documented_no_op(tmp_path):
    from noctis.reporting.run_tree import finish_run

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    first = finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    clock.advance(5 * HOUR)
    again = finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    assert first.sealed is True
    assert again.sealed is False  # nothing happened, and the caller is told so
    assert again.completed_utc == first.completed_utc  # the original stamp stands
    assert _record(store.run_dir)["run"]["completed_utc"] == first.completed_utc


def test_finishing_a_live_locked_run_is_refused(tmp_path):
    """A run another engine is working is not one to seal from underneath it."""
    from noctis.reporting.run_tree import RunLockedError, finish_run

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)  # the lock is held: this "process" is still running

    with pytest.raises(RunLockedError) as excinfo:
        finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    assert store.run_id in str(excinfo.value)
    assert _record(store.run_dir)["run"]["status"] != "completed"


def test_finishing_an_unknown_run_is_a_clean_lookup_failure(tmp_path):
    from noctis.reporting.run_tree import RunNotFoundError, finish_run

    with pytest.raises(RunNotFoundError):
        finish_run(
            tmp_path / "runs",
            "20260101T000000Z-nope00",
            clock=FakeClock(),
            election_metric="sharpe",
        )


def test_a_finished_run_refuses_the_next_resume(tmp_path):
    from noctis.reporting.run_tree import RunCompletedError, finish_run

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    finish_run(runs, store.run_id, clock=clock, election_metric="sharpe")

    with pytest.raises(RunCompletedError):
        _open(runs, clock, run_id=store.run_id, resume=True)


# ── the engine: the cap stops through the shutdown path the time limit already uses ────────


def test_the_machine_stops_from_any_phase_once_the_run_limit_is_reached():
    from noctis.engine import MarketClock, Phase, TradingMachine

    clock = MarketClock("XNYS", "America/New_York")
    machine = TradingMachine(clock, run_limit_hours=2.0)
    machine.start(_et(6, 0))  # pre-open Monday → RESEARCH

    machine.tick(_et(7, 30))
    assert machine.state is Phase.RESEARCH  # 1.5h of 2h
    machine.tick(_et(8, 30))

    assert machine.state is Phase.STOPPED
    assert machine.history[-1] is Phase.STOPPED  # the one terminal move, through stop()


def test_the_run_limit_counts_the_runtime_earlier_segments_already_used():
    """The cap bounds the *run*, not the process: a resumed segment starts with the run's history
    behind it, which is what makes "100 research hours, then stop" survive a stop/resume cycle."""
    from noctis.engine import MarketClock, Phase, TradingMachine

    clock = MarketClock("XNYS", "America/New_York")
    machine = TradingMachine(clock, run_limit_hours=2.0, prior_runtime_s=1.5 * HOUR)
    machine.start(_et(6, 0))

    machine.tick(_et(6, 20))
    assert machine.state is Phase.RESEARCH  # 1h50m of the run's 2h
    machine.tick(_et(6, 40))

    assert machine.state is Phase.STOPPED


def test_each_limit_names_itself_and_the_process_limit_still_wins_the_tie():
    from noctis.engine import MarketClock, TradingMachine

    clock = MarketClock("XNYS", "America/New_York")
    uncapped = TradingMachine(clock)
    per_process = TradingMachine(clock, time_limit_hours=1.0)
    per_run = TradingMachine(clock, run_limit_hours=1.0)
    both = TradingMachine(clock, time_limit_hours=1.0, run_limit_hours=1.0)
    for machine in (uncapped, per_process, per_run, both):
        machine.start(_et(6, 0))

    assert uncapped.limit_hit(_et(23, 0)) is None
    assert per_process.limit_hit(_et(6, 30)) is None
    assert per_process.limit_hit(_et(7, 30)) == "time_limit"
    assert per_run.limit_hit(_et(6, 30)) is None
    assert per_run.limit_hit(_et(7, 30)) == "run_limit"
    assert both.limit_hit(_et(7, 30)) == "time_limit"


def test_a_run_limit_leaves_the_process_time_limit_exactly_as_it_was():
    """``time_limit_hours`` stays per-process and untouched: a machine with no run limit answers
    every question about time the way it always did."""
    from noctis.engine import MarketClock, Phase, TradingMachine

    clock = MarketClock("XNYS", "America/New_York")
    machine = TradingMachine(clock, time_limit_hours=1.0)
    machine.start(_et(10, 0))  # TRADING (open)

    assert machine.time_up(_et(10, 30)) is False
    assert machine.time_up(_et(12, 0)) is True
    machine.tick(_et(12, 0))
    assert machine.state is Phase.STOPPED


def _et(hour: int, minute: int) -> datetime:
    from zoneinfo import ZoneInfo

    return datetime(2027, 1, 4, hour, minute, tzinfo=ZoneInfo("America/New_York"))  # a Monday


# ── the runtime: one shutdown path, and the phase seconds it measures ──────────────────────


def _runtime(tmp_path: Path, *, body: str = "", **kwargs):
    """A runtime over a seeded catalog whose three phases are stand-in objects, not real work."""
    from noctis.config import load_settings
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns
    from noctis.engine import (
        CloseResult,
        ResearchSummary,
        SimulatedSleeper,
        TradingOutcome,
        build_runtime,
    )
    from noctis.memory import MemoryStore

    from ._data_helpers import MockVendor

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\nuniverse: [AAPL, MSFT]\n"
        "session:\n  calendar: XNYS\n  timezone: America/New_York\n"
        f"data:\n  lake_dir: {tmp_path}/lake\n  dataset: EQUS.MINI\n"
        f"state_dir: {tmp_path}/state/\n{textwrap.dedent(body)}"
    )
    lake = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )

    class _Clock(SimulatedSleeper):
        """A controllable clock that *reports* as real-time pacing, so the loop fills the closed
        market with back-to-back research exactly as production does — at CPU speed."""

        wall_clock = True

    captured: dict = {}

    def _factory(start):
        captured["sleeper"] = _Clock(start)
        return captured["sleeper"]

    runtime = build_runtime(
        load_settings(config_path=cfg),
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=_factory,
        **kwargs,
    )

    class _FakeResearch:
        """A stand-in RESEARCH phase: a 20-minute session per entry, no real research."""

        def run(self, panel):
            captured["sleeper"].advance(20 * 60)
            return ResearchSummary()

    class _FakeTrading:
        """A stand-in TRADING phase: a 30-minute session per entry, settling nothing."""

        def run(self, t, sleeper, bars):
            sleeper.advance(30 * 60)
            return TradingOutcome()

    class _FakeClose:
        """A stand-in CLOSE phase: a minute of upkeep, no report on disk."""

        def run(self, t, cycle, *, tracked=None):
            captured["sleeper"].advance(60)
            return CloseResult()

    # The loop holds one object per phase, so the test swaps whole phases — never a method on
    # one — and keeps its own cycle bookkeeping around them.
    runtime.research = _FakeResearch()
    runtime.trading = _FakeTrading()
    runtime.close = _FakeClose()
    return runtime, captured


def test_the_run_limit_stops_the_loop_between_phases_and_names_itself(tmp_path):
    """The cap rides the shutdown path ``time_limit_hours`` already uses — the machine stops
    between phases, the loop leaves through the same exit, and the reason says which limit fired."""
    from noctis.engine import Phase

    runtime, _ = _runtime(tmp_path, body="run_limit_hours: 2\n")

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=_NEVER_REACHED)

    assert result.stopped_reason == "run_limit"
    assert result.history[-1] is Phase.STOPPED
    assert result.phase_seconds["RESEARCH"] > 0


def test_a_resumed_segment_inherits_the_runtime_the_run_already_spent(tmp_path):
    """Two hours of cap with 1h55m already spent leaves minutes, not hours — the segment stops
    almost at once, where a fresh run under the same cap would work for hours."""
    runtime, _ = _runtime(tmp_path, body="run_limit_hours: 2\n", prior_runtime_s=1.9 * HOUR)

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=_NEVER_REACHED)

    assert result.stopped_reason == "run_limit"
    assert result.phase_seconds["RESEARCH"] <= 1200.0  # one 20-minute session, then out


def test_a_run_below_its_cap_stops_for_its_own_reason_and_stays_resumable(tmp_path):
    runtime, _ = _runtime(tmp_path, body="run_limit_hours: 100\n")

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=1)

    assert result.stopped_reason == "max_cycles"


def test_the_process_time_limit_still_stops_a_run_that_has_no_cap_at_all(tmp_path):
    runtime, _ = _runtime(tmp_path, body="time_limit_hours: 1\n")

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=_NEVER_REACHED)

    assert result.stopped_reason == "time_limit"


def test_an_expired_run_limit_never_waits_out_the_closed_market(tmp_path):
    """The between-phase waits are clamped to the *earlier* of the two deadlines, so a run that
    has spent its cap stops instead of pacing to Monday's open."""
    runtime, captured = _runtime(tmp_path, body="run_limit_hours: 1\n", prior_runtime_s=10 * HOUR)

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=_NEVER_REACHED)

    assert result.stopped_reason == "run_limit"
    assert captured["sleeper"].now() <= _et(6, 30).astimezone(UTC)


def test_the_runtime_measures_the_seconds_it_spends_in_each_phase(tmp_path):
    """What the record's cumulative research/trading seconds are summed from. Work, not waiting:
    a phase's seconds are the time its body took, never the pacing between phases."""
    runtime, _ = _runtime(tmp_path, body="run_limit_hours: 100\n")

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=1)

    assert result.phase_seconds["TRADING"] == 1800.0  # one 30-minute session
    assert result.phase_seconds["CLOSE"] == 60.0
    assert result.phase_seconds["RESEARCH"] >= 1200.0  # at least one 20-minute session


# ── engine + store together: the cap seals the run it stopped ──────────────────────────────


def test_a_segment_that_spends_the_cap_closes_completed_and_refuses_the_next_resume(tmp_path):
    """The whole story in one pass, wired the way the CLI wires it: the loop stops between phases
    for ``run_limit``, the segment closes with that reason and its phase timings, and the record —
    deriving the breach from its own segments — is terminal."""
    from noctis.bootstrap import segment_counters, segment_phase_seconds
    from noctis.reporting.run_tree import RunCompletedError

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_capped_inputs(tmp_path / "cfg", 2))
    runtime, _ = _runtime(
        tmp_path / "engine", body="run_limit_hours: 2\n", prior_runtime_s=store.prior_runtime_s
    )

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=_NEVER_REACHED)
    clock.advance(2 * HOUR)  # the wall-clock time this segment really took
    store.close(
        reason=result.stopped_reason,
        counters=segment_counters(result),
        phase_seconds=segment_phase_seconds(result),
    )

    record = _record(store.run_dir)
    assert result.stopped_reason == "run_limit"
    assert record["segments"][0]["stopped_reason"] == "run_limit"
    assert record["segments"][0]["phase_seconds"]["RESEARCH"] > 0
    assert record["run"]["status"] == "completed"
    assert record["run"]["cumulative_research_s"] > 0
    with pytest.raises(RunCompletedError):
        _open(runs, clock, run_id=store.run_id, resume=True)


def test_a_segment_that_stops_below_the_cap_leaves_a_resumable_run(tmp_path):
    from noctis.bootstrap import segment_counters, segment_phase_seconds

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_capped_inputs(tmp_path / "cfg", 100))
    runtime, _ = _runtime(
        tmp_path / "engine", body="run_limit_hours: 100\n", prior_runtime_s=store.prior_runtime_s
    )

    result = runtime.run(start=_et(6, 0).astimezone(UTC), max_cycles=1)
    clock.advance(6 * HOUR)
    store.close(
        reason=result.stopped_reason,
        counters=segment_counters(result),
        phase_seconds=segment_phase_seconds(result),
    )

    record = _record(store.run_dir)
    assert result.stopped_reason == "max_cycles"
    assert record["run"]["status"] == "stopped"
    resumed = _open(runs, clock, run_id=store.run_id, resume=True)
    assert resumed.prior_runtime_s == 6 * HOUR


# ── the CLI ────────────────────────────────────────────────────────────────────────────────


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _config(tmp_path: Path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}")
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _run_ids(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in _runs_dir(tmp_path).iterdir() if p.is_dir())


def _cli_record(tmp_path: Path, run_id: str) -> dict:
    return _read(_runs_dir(tmp_path) / run_id / "run.json")


def test_noctis_run_freezes_the_cap_given_at_creation_into_the_record(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path), "--run-limit-hours", "100"])

    assert result.exit_code == 0, result.output
    record = _cli_record(tmp_path, _run_ids(tmp_path)[0])
    assert record["run"]["run_limit_hours"] == 100.0
    assert record["inputs"]["settings"]["resolved"]["run_limit_hours"] == 100.0
    assert "100" in result.output  # the operator is told what bounds the run they just started


def test_the_cap_cannot_be_moved_on_a_resume_and_the_refusal_says_why(tmp_path):
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg, "--run-limit-hours", "5"]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_id, "--run-limit-hours", "50"]
    )

    assert result.exit_code != 0
    assert "frozen" in result.output
    assert _cli_record(tmp_path, run_id)["run"]["run_limit_hours"] == 5.0


def test_editing_the_cap_in_config_between_segments_does_not_move_it(tmp_path):
    """The cap is frozen tier: it says what compute this experiment was given, so an edit made
    tomorrow cannot retroactively redefine the experiment."""
    cfg = _config(tmp_path, "run_limit_hours: 100\n")
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]
    _config(tmp_path, "run_limit_hours: 1\n")  # the operator edits config.yaml overnight

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert result.exit_code == 0, result.output
    record = _cli_record(tmp_path, run_id)
    assert record["run"]["run_limit_hours"] == 100.0
    assert record["inputs"]["settings"]["resolved"]["run_limit_hours"] == 100.0
    assert record["run"]["status"] == "stopped"  # nowhere near the frozen cap


def test_noctis_run_finish_seals_a_run_without_running_a_segment(tmp_path):
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]
    before = _cli_record(tmp_path, run_id)

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish"])

    assert result.exit_code == 0, result.output
    record = _cli_record(tmp_path, run_id)
    assert record["run"]["status"] == "completed"
    assert record["run"]["completed_utc"] is not None
    assert record["segments"] == before["segments"]  # no segment was opened
    assert run_id in result.output and "completed" in result.output


def test_finishing_a_run_twice_is_a_documented_no_op(tmp_path):
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]
    runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish"])
    sealed_at = _cli_record(tmp_path, run_id)["run"]["completed_utc"]

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish"])

    assert result.exit_code == 0, result.output
    assert "already" in result.output
    assert _cli_record(tmp_path, run_id)["run"]["completed_utc"] == sealed_at


def test_finish_refuses_the_flags_that_shape_a_segment_it_will_never_run(tmp_path):
    """A flag an operator typed must mean something. ``--finish`` runs no segment, so anything
    that shapes the next one is refused beside it rather than quietly ignored."""
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]

    for flag in (["--rebase-config"], ["--run-limit-hours", "5"], ["--time-limit-hours", "5"]):
        result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish", *flag])
        assert result.exit_code != 0, result.output
        assert flag[0] in result.output
    assert _cli_record(tmp_path, run_id)["run"]["status"] == "stopped"  # nothing was sealed


def test_a_finished_run_refuses_the_next_resume_from_the_cli(tmp_path):
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]
    runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish"])

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert result.exit_code != 0
    assert "completed" in result.output
    assert len(_run_ids(tmp_path)) == 1  # refused, never quietly minted a new run


def test_finish_without_a_run_to_seal_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path), "--finish"])

    assert result.exit_code != 0
    assert "--resume" in result.output
    assert not _runs_dir(tmp_path).exists() or not _run_ids(tmp_path)  # nothing was started


def test_noctis_runs_shows_the_cap_beside_the_runtime_it_bounds(tmp_path):
    cfg = _config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg, "--run-limit-hours", "100"]).exit_code == 0
    run_id = _run_ids(tmp_path)[0]

    result = runner.invoke(app, ["runs", "--config", cfg, "--all"])

    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if run_id in line)
    assert "/100h" in line
