"""The run record's pure layer — ``run_record.build`` and ``schema.validate`` (story #129).

Every test here builds a :class:`RunArtifacts` **in memory** and asserts on the returned dict.
That is the whole point of the I/O boundary the epic draws: ``run_store.collect`` does every read,
``run_record.build`` is a pure function over what it collected, and ``schema.validate`` is a pure
check over what ``build`` produced. No filesystem, no clock, no config reaches this file — the
only disk touch is the committed golden record, which is a fixture, not an input.

The structural purity test (``test_build_reaches_no_io_no_clock_and_no_config``) enforces that by
AST, the way ``tests/test_engine_id.py`` does for the engine fingerprint: a future edit that
reaches for ``datetime.now()`` or opens a file fails here rather than silently making the golden
and equivalence tests expensive.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

from noctis.reporting import schema
from noctis.reporting.run_record import (
    EngineIdentity,
    RecordEvent,
    RunArtifacts,
    SegmentArtifact,
    build,
    mark_interrupted,
    resume_refusal,
    utc_iso,
)

RUN_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/reporting/run_record.py"
SCHEMA_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/reporting/schema.py"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "run_record_golden.json"


# ── in-memory fixtures: a two-segment run, every value stated by hand ──────────────────────


ENGINE = EngineIdentity(
    engine_version=1,
    fingerprint={
        "gates": "f63d47b7b9604ab1",
        "backtest": "3ba3e0bf1c97134f",
        "research": "4baf9dea0c82c8cc",
        "prompts": "14eb169506a6b5aa",
        "profiles": "6803b9d26c63d6ae",
        "seeds": "4826fe7224641eb4",
        "memory_seed": "3337fa2cbf896932",
        "schema": None,
    },
    comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe",
    noctis_version="0.1.0",
)


def _environment(host: str, *, cores: int, degraded: list[str] | None = None) -> dict:
    """One machine's environment block, stated by hand — the shape ``environment.capture`` emits.

    Two of these differ in this file, because the whole reason the block is **per segment** is
    that a run may migrate machines mid-experiment (story #139).
    """
    return {
        "hostname_hash": host,
        "os": {"system": "Linux", "release": "7.0.0-14-generic", "arch": "x86_64"},
        "container": True,
        "cpu": {
            "model": "AMD Ryzen 9 7950X",
            "cores_physical": cores // 2,
            "cores_logical": cores,
            "freq_max_mhz": 5881.0,
        },
        "memory_total_bytes": 67351248896,
        "disk_free_bytes": 412000000000,
        "python": "3.11.9",
        "noctis_version": "0.1.0",
        "git": {
            "commit": "a380d3a4f1c0b9e2d5a6f7c8b9a0d1e2f3a4b5c6",
            "branch": "main",
            "dirty": False,
            "describe": "v0.1.0-42-ga380d3a",
        },
        "lockfile_digest": "sha256:9f2c1d",
        "extras_present": {
            "llm": "0.120.0",
            "data": "0.82.0",
            "research": None,
            "engine": None,
            "hardware": "7.2.2",
        },
        "degraded_seams": degraded if degraded is not None else ["engine", "research"],
    }


SMALL_BOX = _environment("9f2c1d3e4b5a", cores=8)
BIG_BOX = _environment("0011223344ff", cores=32)


def _segment(
    index: int,
    *,
    started: str,
    stopped: str | None,
    reason: str | None,
    status: str,
    resumed: bool = False,
    counters: dict[str, int] | None = None,
    phase_seconds: dict[str, float] | None = None,
    environment: dict | None = None,
) -> SegmentArtifact:
    return SegmentArtifact(
        index=index,
        started_utc=started,
        stopped_utc=stopped,
        stopped_reason=reason,
        status=status,
        argv=("run", "-v"),
        command="run",
        resumed=resumed,
        counters=counters or {"cycles": 1, "research_iterations": 4, "trades": 2},
        phase_seconds=phase_seconds or {"RESEARCH": 12000.0, "TRADING": 2340.0, "CLOSE": 60.0},
        engine=ENGINE,
        environment=environment if environment is not None else SMALL_BOX,
    )


def _artifacts(**overrides) -> RunArtifacts:
    """A closed two-segment run — the shape the golden record snapshots."""
    base = dict(
        run_id="20260727T142233Z-a1b2c3",
        label=None,
        created_utc="2026-07-27T14:22:33.418Z",
        last_active_utc="2026-07-28T02:10:04.002Z",
        completed_utc=None,
        complete=True,
        segments=(
            _segment(
                0,
                started="2026-07-27T14:22:33.418Z",
                stopped="2026-07-27T18:22:33.418Z",
                reason="time_limit",
                status="stopped",
            ),
            _segment(
                1,
                started="2026-07-28T01:10:04.002Z",
                stopped="2026-07-28T02:10:04.002Z",
                reason="stop_requested",
                status="stopped",
                resumed=True,
                counters={"cycles": 2, "research_iterations": 9, "trades": 0},
                phase_seconds={"RESEARCH": 3400.0, "TRADING": 0.0, "CLOSE": 40.0},
                # The run migrated machines between the two nights — the case the per-segment
                # environment block exists for.
                environment=BIG_BOX,
            ),
        ),
        engine=ENGINE,
        events=(
            RecordEvent(
                t="2026-07-28T01:10:04.002Z",
                segment=1,
                kind="warn",
                text="stole a stale run lock (pid 4242 is not running on this host)",
            ),
        ),
        errors=(),
    )
    base.update(overrides)
    return RunArtifacts(**base)  # type: ignore[arg-type]


# ── the record's declared shape ────────────────────────────────────────────────────────────


def test_a_record_declares_schema_version_one_and_kind_noctis_run():
    record = build(_artifacts())

    assert record["schema_version"] == 1
    assert record["kind"] == "noctis.run"
    assert schema.SCHEMA_VERSION == 1
    assert schema.KIND == "noctis.run"


def test_the_record_carries_identity_lifecycle_segments_engine_and_events_sections():
    record = build(_artifacts())

    assert set(record) >= {"run", "segments", "engine", "events", "errors"}
    assert record["run"]["run_id"] == "20260727T142233Z-a1b2c3"
    assert record["run"]["created_utc"] == "2026-07-27T14:22:33.418Z"
    assert record["run"]["last_active_utc"] == "2026-07-28T02:10:04.002Z"
    assert record["run"]["status"] == "stopped"
    assert len(record["segments"]) == 2
    assert record["engine"]["engine_version"] == 1
    assert record["engine"]["fingerprint"]["gates"] == "f63d47b7b9604ab1"
    assert record["engine"]["comparable_key"] == "1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe"
    assert record["events"][0]["kind"] == "warn"


def test_every_record_carries_an_engine_identity_even_with_a_null_component():
    """A record without an engine identity can never be retrofitted into a comparison bucket,
    so the section is mandatory — a null *component* is fine, a missing section is not."""
    record = build(_artifacts())

    assert set(record["engine"]["fingerprint"]) == set(ENGINE.fingerprint)
    assert record["engine"]["fingerprint"]["schema"] is None  # explicit null, never omitted
    assert record["segments"][0]["engine_version"] == 1
    assert record["segments"][0]["engine_fingerprint"]["gates"] == "f63d47b7b9604ab1"


# ── the frozen inputs (story #132) ─────────────────────────────────────────────────────────


FROZEN_INPUTS = {
    "config_epoch": 1,
    # Empty until an operator deliberately adopts a changed configuration (story #134): the epoch
    # and its change log live together, because one is meaningless without the other.
    "config_changes": [],
    "frozen_at_utc": "2026-07-27T14:22:33.418Z",
    "execution_mode": "paper",
    "mandate": {"source": "profile:aggressive", "text": "Trade volatility.", "text_sha256": "ab"},
    # The rest of the provenance block (story #139) — derived views over what ``resolved`` already
    # carries, so a consumer reads "what produced these numbers" without rebuilding a fallback
    # chain. A model name is public; the key that authenticates it is secret tier and never here.
    "models": {
        "research": "openai/gpt-5.4",
        "coder": None,
        "coder_fallback": None,
        "ideation": "claude-opus-4-8",
        "research_loop": "conversation",
        "context_window": None,
        "cost_profile": "balanced",
    },
    "data": {
        "provider": "databento",
        "dataset": "EQUS.MINI",
        "lake_dir": "workspace/data_lake",
    },
    "settings": {
        "digest": "0f1e2d3c4b5a",
        "resolved": {"promotion": {"metric": "sharpe"}},
        "frozen_keys": ["promotion.metric"],
        "live_keys": ["workspace_dir"],
        "refused_keys": ["allow_live", "mode"],
    },
}


def test_the_frozen_inputs_are_carried_verbatim_and_never_reinterpreted():
    """The builder is free of configuration on purpose: freezing policy lives in
    ``config.rehydrate``, and this module round-trips the block it is handed."""
    record = build(_artifacts(inputs=FROZEN_INPUTS))

    assert record["inputs"] == FROZEN_INPUTS
    assert mark_interrupted(_artifacts(inputs=FROZEN_INPUTS)).inputs == FROZEN_INPUTS


def test_a_run_that_froze_no_config_says_so_with_an_explicit_null():
    assert build(_artifacts())["inputs"] is None
    assert "inputs" in build(_artifacts())


def test_the_validator_refuses_a_record_carrying_either_live_money_gate():
    """A record may never offer a second source for a decision that must have two independent
    ones (AGENTS.md rule 1) — so carrying one is a schema violation, not a detail."""
    inputs = json.loads(json.dumps(FROZEN_INPUTS))
    inputs["settings"]["resolved"]["mode"] = "live"
    inputs["settings"]["resolved"]["allow_live"] = True

    problems = schema.validate(build(_artifacts(inputs=inputs)))

    assert any("mode" in problem for problem in problems)
    assert any("allow_live" in problem for problem in problems)


def test_the_validator_names_a_frozen_inputs_block_missing_a_key():
    inputs = json.loads(json.dumps(FROZEN_INPUTS))
    del inputs["settings"]["digest"]
    del inputs["execution_mode"]

    problems = schema.validate(build(_artifacts(inputs=inputs)))

    assert any("inputs.settings.digest" in problem for problem in problems)
    assert any("inputs.execution_mode" in problem for problem in problems)


def test_validate_accepts_a_record_with_frozen_inputs():
    assert schema.validate(build(_artifacts(inputs=FROZEN_INPUTS))) == []


def test_validate_accepts_a_record_whose_config_was_rebased_mid_run():
    """A run that adopted a new configuration mid-flight is a valid record — a *marked* one."""
    inputs = json.loads(json.dumps(FROZEN_INPUTS))
    inputs["config_epoch"] = 2
    inputs["config_changes"] = [_CONFIG_CHANGE]

    assert schema.validate(build(_artifacts(inputs=inputs))) == []


def test_the_validator_names_a_config_change_that_does_not_say_where_it_happened():
    """A mid-run config change must say so *and say where*, or a reader cannot tell which segments
    ran under which configuration — and every comparison built on the record is then guesswork."""
    inputs = json.loads(json.dumps(FROZEN_INPUTS))
    change = json.loads(json.dumps(_CONFIG_CHANGE))
    del change["segment"]
    change["at"] = "yesterday"
    inputs["config_changes"] = [change]

    problems = schema.validate(build(_artifacts(inputs=inputs)))

    assert any("inputs.config_changes[0].segment" in problem for problem in problems)
    assert any("inputs.config_changes[0].at" in problem for problem in problems)


_CONFIG_CHANGE = {
    "at": "2026-08-03T02:10:04.002Z",
    "segment": 4,
    "from_epoch": 1,
    "to_epoch": 2,
    "digest_before": "0f1e2d3c4b5a",
    "digest_after": "5a4b3c2d1e0f",
    "settings": [{"path": "promotion.metric", "from": "sharpe", "to": "sortino"}],
    "mandate": None,
}


# ── resuming is refused only where it must be ──────────────────────────────────────────────


def test_only_a_completed_run_refuses_another_segment():
    """``running`` is the shape a *crash* leaves behind as well as the shape a live engine
    writes; telling those apart is the liveness lock's job, not a status field's."""
    for status in ("stopped", "interrupted", "running"):
        assert resume_refusal({"run": {"status": status}}) is None
    assert resume_refusal({}) is None

    refusal = resume_refusal({"run": {"status": "completed"}})

    assert refusal is not None and "completed" in refusal


def test_absent_values_are_explicit_nulls_not_omitted_keys():
    record = build(_artifacts(label=None, completed_utc=None))

    assert record["run"]["label"] is None
    assert record["run"]["completed_utc"] is None
    assert "label" in record["run"] and "completed_utc" in record["run"]


def test_timestamps_are_utc_iso_8601_with_a_z_suffix():
    stamp = utc_iso(datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC))

    assert stamp == "2026-07-27T14:22:33.418Z"
    record = build(_artifacts())
    assert record["run"]["created_utc"].endswith("Z")
    assert record["segments"][0]["started_utc"].endswith("Z")


# ── segments: one per process invocation ───────────────────────────────────────────────────


def test_each_segment_carries_start_stop_duration_reason_argv_and_its_own_counters():
    record = build(_artifacts())

    first, second = record["segments"]
    assert first["index"] == 0 and second["index"] == 1
    assert first["started_utc"] == "2026-07-27T14:22:33.418Z"
    assert first["stopped_utc"] == "2026-07-27T18:22:33.418Z"
    assert first["duration_s"] == 14400.0
    assert first["stopped_reason"] == "time_limit"
    assert first["argv"] == ["run", "-v"]
    assert first["command"] == "run"
    assert first["resumed"] is False and second["resumed"] is True
    assert first["counters"] == {"cycles": 1, "research_iterations": 4, "trades": 2}
    assert second["counters"] == {"cycles": 2, "research_iterations": 9, "trades": 0}


def test_an_open_segment_reports_a_null_stop_stamp_and_null_duration():
    artifacts = _artifacts(
        segments=(
            _segment(
                0, started="2026-07-27T14:22:33.418Z", stopped=None, reason=None, status="running"
            ),
        ),
    )

    record = build(artifacts)

    assert record["segments"][0]["stopped_utc"] is None
    assert record["segments"][0]["duration_s"] is None
    assert record["run"]["status"] == "running"


def test_cumulative_runtime_is_the_sum_of_the_closed_segment_durations():
    record = build(_artifacts())

    assert record["run"]["cumulative_runtime_s"] == 14400.0 + 3600.0


def test_segments_are_uncapped_they_are_the_runs_spine():
    many = tuple(
        _segment(
            i,
            started="2026-07-27T14:22:33.418Z",
            stopped="2026-07-27T14:22:34.418Z",
            reason="time_limit",
            status="stopped",
        )
        for i in range(3000)
    )

    record = build(_artifacts(segments=many))

    assert len(record["segments"]) == 3000
    assert record["run"]["truncated"] == {}


# ── the environment is per segment, never per run (story #139) ─────────────────────────────


def test_each_segment_carries_the_environment_it_actually_ran_in():
    record = build(_artifacts())

    first, second = record["segments"]
    assert first["environment"] == SMALL_BOX
    assert second["environment"] == BIG_BOX


def test_a_run_whose_segments_ran_on_different_machines_shows_different_environments():
    """The whole reason the block is per segment: research throughput is CPU-bound, so one
    segment's trials-per-hour must never be attributed to another segment's hardware."""
    record = build(_artifacts())

    first, second = record["segments"]
    assert first["environment"]["hostname_hash"] != second["environment"]["hostname_hash"]
    assert first["environment"]["cpu"]["cores_logical"] == 8
    assert second["environment"]["cpu"]["cores_logical"] == 32
    assert schema.validate(record) == []


def test_the_record_also_exposes_the_latest_environment():
    """One fetch, one page: a consumer showing "the machine this run is on" reads one key rather
    than walking segments backwards looking for the newest non-null one."""
    record = build(_artifacts())

    assert record["environment_latest"] == BIG_BOX
    assert record["environment_latest"] == record["segments"][-1]["environment"]


def test_the_latest_environment_skips_a_segment_that_measured_none():
    """A record written before environments existed (or by a store handed none) still answers
    "what machine has this run been on?" with the newest one that *was* measured."""
    artifacts = _artifacts(
        segments=(
            _segment(
                0,
                started="2026-07-27T14:22:33.418Z",
                stopped="2026-07-27T18:22:33.418Z",
                reason="time_limit",
                status="stopped",
            ),
            _segment(
                1,
                started="2026-07-28T01:10:04.002Z",
                stopped=None,
                reason=None,
                status="running",
                environment={},
            ),
        )
    )

    record = build(artifacts)

    assert record["segments"][1]["environment"] is None
    assert record["environment_latest"] == SMALL_BOX


def test_a_run_that_measured_no_environment_at_all_says_so_with_an_explicit_null():
    artifacts = _artifacts(
        segments=(SegmentArtifact(index=0, started_utc="2026-07-27T14:22:33.418Z", engine=ENGINE),)
    )

    record = build(artifacts)

    assert record["segments"][0]["environment"] is None
    assert record["environment_latest"] is None
    assert "environment_latest" in record  # an explicit null, never an omitted key
    assert schema.validate(record) == []


def test_the_validator_names_an_environment_block_missing_a_key():
    """Absent values are explicit nulls here too — a silently missing key would be
    indistinguishable from "this schema version did not record it"."""
    record = build(_artifacts())
    del record["segments"][0]["environment"]["degraded_seams"]
    del record["environment_latest"]["git"]

    problems = schema.validate(record)

    assert any("degraded_seams" in problem for problem in problems)
    assert any("environment_latest.git" in problem for problem in problems)


def test_the_record_stays_valid_when_every_environment_fact_degraded():
    """The degradation case end to end: no git, no psutil, no extras — schema-valid, explicit
    nulls, and the seams that were missing named."""
    degraded = {
        "hostname_hash": None,
        "os": {"system": "Linux", "release": None, "arch": None},
        "container": None,
        "cpu": dict.fromkeys(("model", "cores_physical", "cores_logical", "freq_max_mhz")),
        "memory_total_bytes": None,
        "disk_free_bytes": None,
        "python": "3.11.9",
        "noctis_version": None,
        "git": None,
        "lockfile_digest": None,
        "extras_present": dict.fromkeys(("llm", "data", "research", "engine", "hardware")),
        "degraded_seams": ["data", "engine", "git", "hardware", "llm", "lockfile", "research"],
    }

    record = build(
        _artifacts(
            segments=(
                _segment(
                    0,
                    started="2026-07-27T14:22:33.418Z",
                    stopped="2026-07-27T18:22:33.418Z",
                    reason="time_limit",
                    status="stopped",
                    environment=degraded,
                ),
            )
        )
    )

    assert schema.validate(record) == []
    assert record["environment_latest"] == degraded
    assert "git" in record["environment_latest"]["degraded_seams"]


# ── a research-only run is a first-class shape, not a degenerate one (story #137) ───────────


def _research_segment(index: int, *, resumed: bool, research_s: float) -> SegmentArtifact:
    """One ``noctis research --resume`` segment: a research-only night, no trading at all."""
    return SegmentArtifact(
        index=index,
        started_utc="2026-07-27T14:22:33.418Z",
        stopped_utc="2026-07-27T18:22:33.418Z",
        stopped_reason="agent_done",
        status="stopped",
        argv=("research", "--resume", "latest"),
        command="research",
        resumed=resumed,
        counters={"sessions": 1, "research_iterations": 6, "research_promotions": 0},
        phase_seconds={"RESEARCH": research_s},
        engine=ENGINE,
    )


def test_a_research_only_segment_says_so_through_the_command_it_was_run_by():
    """Research-only is *derived*, not a second flag: the segment records the verb that produced
    it, and ``research`` is a verb that cannot trade."""
    record = build(_artifacts(segments=(_research_segment(0, resumed=False, research_s=900.0),)))

    segment = record["segments"][0]
    assert segment["command"] == "research"
    assert segment["argv"] == ["research", "--resume", "latest"]
    assert segment["stopped_reason"] == "agent_done"
    assert segment["counters"]["research_iterations"] == 6
    assert schema.validate(record) == []


def test_a_run_that_only_ever_researched_reports_traded_false_and_null_performance():
    """Epic D10 / §5.6: a website must render "researching", never a fake flat 0% equity curve."""
    record = build(
        _artifacts(
            segments=(
                _research_segment(0, resumed=False, research_s=900.0),
                _research_segment(1, resumed=True, research_s=100.0),
            )
        )
    )

    assert record["run"]["traded"] is False
    assert record["performance"] is None
    assert "performance" in record  # an explicit null, never an omitted key
    assert record["run"]["cumulative_research_s"] == 1000.0
    assert record["run"]["cumulative_trading_s"] == 0.0


def test_a_run_whose_segments_recorded_trades_reports_traded_true():
    assert build(_artifacts())["run"]["traded"] is True


def test_research_only_segments_carry_their_seconds_into_the_runs_research_total():
    """A research-only night belongs to the same run: its hours accumulate beside the loop's."""
    mixed = _artifacts(
        segments=(
            _segment(
                0,
                started="2026-07-27T14:22:33.418Z",
                stopped="2026-07-27T18:22:33.418Z",
                reason="time_limit",
                status="stopped",
            ),
            _research_segment(1, resumed=True, research_s=600.0),
        )
    )

    record = build(mixed)

    assert record["run"]["cumulative_research_s"] == 12000.0 + 600.0
    assert record["run"]["traded"] is True  # the loop segment traded; the research one did not


def test_the_validator_refuses_a_performance_block_on_a_run_that_never_traded():
    """The rule story #142 has to keep: no trading, no performance — not even zeros."""
    record = build(_artifacts(segments=(_research_segment(0, resumed=False, research_s=900.0),)))
    record["performance"] = {"returns": {"total_return_pct": 0.0}}

    problems = schema.validate(record)

    assert any("performance" in problem and "traded" in problem for problem in problems)


def test_the_validator_names_a_segment_run_by_an_unknown_command():
    record = build(_artifacts())
    record["segments"][0]["command"] = "meditate"

    assert any("command" in problem for problem in schema.validate(record))


def test_the_run_reports_the_trials_it_journaled_or_an_explicit_null():
    """The multiple-testing count DSR needs, read off the run's own experiment journals — never a
    counter incremented in memory. ``null`` when nothing was journaled at all."""
    assert build(_artifacts(trials=3180))["run"]["cumulative_trials"] == 3180
    assert build(_artifacts())["run"]["cumulative_trials"] is None
    assert "cumulative_trials" in build(_artifacts())["run"]


# ── interrupted is decided on the next OPEN, never guessed at write time ────────────────────


def test_a_segment_with_no_stop_stamp_is_marked_interrupted_on_the_next_open():
    killed = _artifacts(
        segments=(
            _segment(
                0, started="2026-07-27T14:22:33.418Z", stopped=None, reason=None, status="running"
            ),
        ),
    )

    reopened = mark_interrupted(killed)

    assert build(reopened)["segments"][0]["status"] == "interrupted"
    assert build(reopened)["run"]["status"] == "interrupted"
    # …and the write-time record still says "running": the kill is not guessed at, it is
    # observed on the next open.
    assert build(killed)["segments"][0]["status"] == "running"


def test_marking_interrupted_leaves_cleanly_closed_segments_alone():
    reopened = mark_interrupted(_artifacts())

    assert [s["status"] for s in build(reopened)["segments"]] == ["stopped", "stopped"]
    assert build(reopened)["run"]["status"] == "stopped"


# ── caps: every cap that bites says so ─────────────────────────────────────────────────────


def test_the_event_cap_writes_a_truncation_note_with_kept_and_total_counts():
    total = schema.EVENT_CAP + 25
    events = tuple(
        RecordEvent(t="2026-07-27T14:22:33.418Z", segment=0, kind="info", text=f"event {i}")
        for i in range(total)
    )

    record = build(_artifacts(events=events))

    assert len(record["events"]) == schema.EVENT_CAP
    assert record["run"]["truncated"]["events"] == {"kept": schema.EVENT_CAP, "total": total}


def test_an_uncapped_run_carries_an_empty_truncation_block():
    record = build(_artifacts())

    assert record["run"]["truncated"] == {}


# ── determinism ────────────────────────────────────────────────────────────────────────────


def test_building_the_same_artifacts_twice_returns_identical_records():
    assert build(_artifacts()) == build(_artifacts())


def test_two_runs_differing_only_in_identity_and_stamps_produce_identical_records():
    """Determinism (the epic's test): same seed, config and injected clock ⇒ identical records
    except identity and timestamps. Mask those and the two records are byte-identical."""
    first = build(_artifacts())
    second = build(
        _artifacts(
            run_id="20260901T090000Z-ffffff",
            created_utc="2026-09-01T09:00:00.000Z",
            last_active_utc="2026-09-01T21:00:00.000Z",
        )
    )

    assert _masked(first) == _masked(second)
    assert first != second  # the masking is doing real work


def _masked(record: dict) -> str:
    masked = json.loads(json.dumps(record))
    masked["run"]["run_id"] = "<id>"
    masked["run"]["created_utc"] = "<t>"
    masked["run"]["last_active_utc"] = "<t>"
    return json.dumps(masked, sort_keys=True)


# ── the golden record ──────────────────────────────────────────────────────────────────────


def test_the_golden_record_still_matches_the_committed_fixture():
    """Snapshot the whole record so schema drift is visible in review, not discovered by a
    website. Regenerate deliberately (and explain the diff) when the schema grows."""
    assert build(_artifacts()) == json.loads(GOLDEN.read_text())


def test_the_golden_record_is_schema_valid():
    assert schema.validate(json.loads(GOLDEN.read_text())) == []


# ── the schema check ───────────────────────────────────────────────────────────────────────


def test_validate_accepts_every_record_build_produces():
    assert schema.validate(build(_artifacts())) == []
    assert schema.validate(build(mark_interrupted(_artifacts()))) == []


def test_validate_names_a_wrong_schema_version_and_kind():
    record = build(_artifacts())
    record["schema_version"] = 2
    record["kind"] = "noctis.something-else"

    problems = schema.validate(record)

    assert any("schema_version" in p for p in problems)
    assert any("kind" in p for p in problems)


def test_validate_names_a_missing_section_and_an_unknown_status():
    record = build(_artifacts())
    del record["segments"]
    record["run"]["status"] = "finished"

    problems = schema.validate(record)

    assert any("segments" in p for p in problems)
    assert any("status" in p for p in problems)


def test_validate_names_a_timestamp_without_a_z_suffix():
    record = build(_artifacts())
    record["run"]["created_utc"] = "2026-07-27 14:22:33"

    assert any("created_utc" in p for p in schema.validate(record))


def test_validate_names_an_out_of_order_segment_index():
    record = build(_artifacts())
    record["segments"][1]["index"] = 7

    assert any("index" in p for p in schema.validate(record))


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_build_reaches_no_io_no_clock_and_no_config():
    """The load-bearing decision of this slice: ``build`` is pure. Timestamps arrive as data
    inside ``RunArtifacts``; nothing here may read a clock, a file or the settings."""
    assert _imports(RUN_RECORD_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "typing",
        "noctis",
    }
    text = RUN_RECORD_SOURCE.read_text()
    for forbidden in ("datetime.now", "utcnow", "open(", "Path(", "os.", "json."):
        assert forbidden not in text, forbidden


def test_the_schema_module_is_pure_stdlib():
    assert _imports(SCHEMA_SOURCE) <= {"__future__", "collections", "typing"}
