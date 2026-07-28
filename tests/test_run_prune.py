"""Retention: opt-in pruning of a *completed* run's heavy state (story #138, epic #126).

The run tree grows: `state/`, `strategies/` and `reports/` are megabytes a night, while `run.json`
and `index.json` are kilobytes and *are* the long-term progress history the record exists for. So
retention reclaims the first three and never touches the last two — and it does so **only** for
runs in status `completed`, because pruning a `stopped` or `interrupted` run's state would
silently destroy its resumability, the one thing this design promises.

Everything asserted here is external: what is on disk after the call, what the record says, what
the CLI prints, what the next resume does. Every deletion happens inside ``tmp_path``.
"""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.reporting import schema
from noctis.reporting.run_record import (
    PRUNABLE_STATUSES,
    TERMINAL_STATUSES,
    EngineIdentity,
    RunArtifacts,
    SegmentArtifact,
    build,
    prune_refusal,
    resume_refusal,
)

runner = CliRunner()

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
HOUR = 3600.0

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


def _record_with(status: str) -> dict:
    """A record in one of the four lifecycle statuses, built the way a real write builds it."""
    open_segment = SegmentArtifact(index=0, started_utc=_stamp(0), engine=ENGINE, status="running")
    closed = SegmentArtifact(
        index=0,
        started_utc=_stamp(0),
        stopped_utc=_stamp(HOUR),
        stopped_reason="stop_requested",
        status="stopped",
        engine=ENGINE,
    )
    interrupted = SegmentArtifact(
        index=0, started_utc=_stamp(0), status="interrupted", engine=ENGINE
    )
    segment = {"running": open_segment, "stopped": closed, "interrupted": interrupted}.get(
        status, closed
    )
    artifacts = RunArtifacts(
        run_id="20260727T142233Z-a1b2c3",
        created_utc=_stamp(0),
        last_active_utc=_stamp(HOUR),
        engine=ENGINE,
        segments=(segment,),
        completed_utc=_stamp(2 * HOUR) if status == "completed" else None,
    )
    record = build(artifacts)
    assert record["run"]["status"] == status  # the fixture is what it claims to be
    return record


# ── the pure rule: prunable is the exact complement of resumable ───────────────────────────


def test_a_run_may_be_pruned_exactly_when_it_may_never_gain_another_segment():
    """The proof that a pruned-then-resumed run is impossible, made structurally rather than by a
    second guard: for every status the schema allows, exactly one of the two refusals is silent."""
    for status in schema.RUN_STATUSES:
        record = _record_with(status)
        prunable = prune_refusal(record) is None
        resumable = resume_refusal(record) is None
        assert prunable is not resumable, status


def test_the_prunable_statuses_are_the_terminal_ones_by_construction():
    """One constant, not two: a status that stopped being terminal would stop being prunable in
    the same edit, so the two rules can never drift into a window where both are true."""
    assert PRUNABLE_STATUSES is TERMINAL_STATUSES
    assert PRUNABLE_STATUSES == ("completed",)


@pytest.mark.parametrize("status", ["stopped", "interrupted", "running"])
def test_pruning_a_resumable_run_is_refused_because_resumability_would_be_destroyed(status):
    refusal = prune_refusal(_record_with(status))

    assert refusal is not None
    assert status in refusal
    assert "resum" in refusal  # the refusal names the thing pruning would destroy
    assert "completed" in refusal  # …and the one status that may be pruned


def test_a_completed_run_may_be_pruned_and_says_nothing_about_it():
    assert prune_refusal(_record_with("completed")) is None


def test_a_record_with_no_readable_run_section_is_never_prunable():
    """A record we cannot read the status off is not evidence that pruning is safe."""
    assert prune_refusal({}) is not None
    assert prune_refusal({"run": "not-an-object"}) is not None


# ── the record marks the prune, and stays schema-valid ─────────────────────────────────────


def test_the_record_carries_state_pruned_and_it_is_false_until_something_prunes():
    record = _record_with("completed")

    assert record["run"]["state_pruned"] is False
    assert schema.validate(record) == []


def test_a_pruned_record_is_still_schema_valid():
    artifacts = RunArtifacts(
        run_id="20260727T142233Z-a1b2c3",
        created_utc=_stamp(0),
        last_active_utc=_stamp(HOUR),
        engine=ENGINE,
        completed_utc=_stamp(HOUR),
        state_pruned=True,
    )

    record = build(artifacts)

    assert record["run"]["state_pruned"] is True
    assert schema.validate(record) == []


# ── the store: what pruning does to the disk ───────────────────────────────────────────────


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
    from noctis.reporting.run_store import open_run

    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _on_disk(run_dir: Path) -> dict:
    from noctis.reporting.run_store import RUN_RECORD_NAME

    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _fill(run_dir: Path) -> dict[str, int]:
    """Give a run the heavy tree a real night leaves behind; return the bytes per directory."""
    written = {
        "state": [("experiments/momentum.jsonl", "x" * 700), ("champions.json", "y" * 300)],
        "strategies": [("__tmp/draft.py", "z" * 500), ("champions/momo.py", "c" * 250)],
        "reports": [("2026-07-27.md", "r" * 400)],
        # Never named by retention: the run's agent memory is small, and it is history.
        "memory": [("MEMORY.md", "m" * 100)],
    }
    sizes: dict[str, int] = {}
    for folder, files in written.items():
        for relative, body in files:
            path = run_dir / folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        sizes[folder] = sum(len(body) for _, body in files)
    return sizes


def _completed_run(runs: Path, clock: FakeClock, **kwargs) -> Path:
    """One empty run taken all the way to ``completed`` — the only status retention may touch."""
    store = _open(runs, clock, **kwargs)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    _finish(runs, store.run_id, clock)
    return store.run_dir


def _completed_with_state(runs: Path, clock: FakeClock, **kwargs) -> tuple[Path, dict[str, int]]:
    """A completed run that actually did a night's work: the heavy tree, then the seal."""
    store = _open(runs, clock, **kwargs)
    sizes = _fill(store.run_dir)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    _finish(runs, store.run_id, clock)
    return store.run_dir, sizes


def _prune(runs: Path, address: str, clock: FakeClock, **kwargs):
    from noctis.reporting.run_store import prune_run_state

    return prune_run_state(runs, address, clock=clock, election_metric="sharpe", **kwargs)


def test_pruning_a_completed_run_removes_the_heavy_directories_and_keeps_the_record(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, sizes = _completed_with_state(runs, clock)

    outcome = _prune(runs, run_dir.name, clock)

    assert sorted(outcome.removed) == ["reports", "state", "strategies"]
    assert outcome.freed_bytes == sizes["state"] + sizes["strategies"] + sizes["reports"]
    assert not (run_dir / "state").exists()
    assert not (run_dir / "strategies").exists()
    assert not (run_dir / "reports").exists()
    # …and everything retention does not name is untouched, the record above all.
    assert (run_dir / "run.json").is_file()
    assert (runs / "index.json").is_file()
    assert (run_dir / "memory" / "MEMORY.md").read_text() == "m" * 100


def test_pruning_sets_state_pruned_on_the_record(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    assert _on_disk(run_dir)["run"]["state_pruned"] is False

    _prune(runs, run_dir.name, clock)

    record = _on_disk(run_dir)
    assert record["run"]["state_pruned"] is True
    assert schema.validate(record) == []


def test_pruning_changes_nothing_in_the_record_but_the_marker_and_a_note(tmp_path):
    """Whatever the record *embeds* survives a prune by construction — pruning touches three
    directories and rewrites one flag. (The embedded champion sources themselves arrive with story
    #141; this is the rule they will land into.)"""
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    before = _on_disk(run_dir)

    _prune(runs, run_dir.name, clock)

    after = _on_disk(run_dir)
    assert after["run"].pop("state_pruned") is True
    assert before["run"].pop("state_pruned") is False
    assert after.pop("events")[:-1] == before.pop("events")  # one appended note, nothing rewritten
    assert after == before


def test_pruning_keeps_the_progress_history_the_record_counted_from_the_pruned_state(tmp_path):
    """The trial count is read off ``state/experiments/*.jsonl`` at every write — so a prune that
    rewrote the record *after* deleting them would erase the run's own history. It is collected
    first, and the number the record already carried stands."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    _journal(store.run_dir, trials=3)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    _finish(runs, store.run_id, clock)
    assert _on_disk(store.run_dir)["run"]["cumulative_trials"] == 3

    _prune(runs, store.run_id, clock)

    record = _on_disk(store.run_dir)
    assert record["run"]["cumulative_trials"] == 3
    assert record["run"]["cumulative_runtime_s"] == HOUR
    assert len(record["segments"]) == 1


def _journal(run_dir: Path, *, trials: int) -> None:
    """Write ``trials`` distinct trial lines into the run's own experiment journal."""
    path = run_dir / "state" / "experiments" / "momentum.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"event": "trial", "params": {"n": n}, "metric": 0.1 * n}) + "\n"
            for n in range(trials)
        )
    )


def _finish(runs: Path, run_id: str, clock: FakeClock) -> None:
    from noctis.reporting.run_store import finish_run

    finish_run(runs, run_id, clock=clock, election_metric="sharpe")


@pytest.mark.parametrize("status", ["stopped", "interrupted", "running"])
def test_pruning_a_run_that_could_still_resume_is_refused_and_removes_nothing(tmp_path, status):
    from noctis.reporting.run_store import RunNotPrunableError

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    if status == "stopped":
        clock.advance(HOUR)
        store.close(reason="stop_requested")
    elif status == "interrupted":
        clock.advance(HOUR)
        store.close(reason="stop_requested")
        _interrupt(store.run_dir)
    sizes = _fill(store.run_dir)
    assert _on_disk(store.run_dir)["run"]["status"] == status

    with pytest.raises(RunNotPrunableError) as excinfo:
        _prune(runs, store.run_id, clock)

    assert store.run_id in str(excinfo.value)
    assert "resum" in str(excinfo.value)
    assert (store.run_dir / "state" / "experiments" / "momentum.jsonl").exists()
    assert sum(_dir_bytes(store.run_dir / name) for name in sizes) == sum(sizes.values())
    assert _on_disk(store.run_dir)["run"]["state_pruned"] is False


def _interrupt(run_dir: Path) -> None:
    """Leave the record in the shape a kill leaves: a segment with no stop stamp."""
    record = _on_disk(run_dir)
    record["segments"][-1].update(stopped_utc=None, status="interrupted", duration_s=None)
    record["run"]["status"] = "interrupted"
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))


def _dir_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def test_a_stopped_run_is_still_resumable_after_a_refused_prune(tmp_path):
    """The refusal is not a technicality: the run it protected still resumes, with its state."""
    from noctis.reporting.run_store import RunNotPrunableError

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(HOUR)
    store.close(reason="stop_requested")
    _fill(store.run_dir)
    with pytest.raises(RunNotPrunableError):
        _prune(runs, store.run_id, clock)

    resumed = _open(runs, clock, run_id=store.run_id, resume=True)

    assert resumed.prior_runtime_s == HOUR
    assert (store.run_dir / "state" / "champions.json").exists()
    resumed.close(reason="stop_requested")


def test_pruning_a_run_another_engine_holds_is_refused(tmp_path):
    """A live lock is the second gate: deleting the directories an engine is reading and writing
    is corruption, whatever the record says about the run's status."""
    from noctis.reporting.run_store import RunLockedError

    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    _hold_lock(run_dir, clock)

    with pytest.raises(RunLockedError) as excinfo:
        _prune(runs, run_dir.name, clock)

    assert run_dir.name in str(excinfo.value)
    assert (run_dir / "state" / "champions.json").exists()
    assert _on_disk(run_dir)["run"]["state_pruned"] is False


def _hold_lock(run_dir: Path, clock: FakeClock) -> None:
    """A lock this very process holds — the one lock no staleness check can dismiss."""
    import hashlib
    import os
    import socket

    from noctis.reporting.run_record import utc_iso

    (run_dir / "run.lock").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "pid": os.getpid(),
                "hostname_hash": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[
                    :12
                ],
                "started_utc": utc_iso(clock()),
                "heartbeat_utc": utc_iso(clock()),
            }
        )
    )


def test_a_pruned_run_can_never_be_resumed(tmp_path):
    """The story's required test — and it needs no new guard: only a ``completed`` run may be
    pruned, and ``completed`` is exactly the status a resume refuses."""
    from noctis.reporting.run_store import RunCompletedError

    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    _prune(runs, run_dir.name, clock)

    with pytest.raises(RunCompletedError):
        _open(runs, clock, run_id=run_dir.name, resume=True)


def test_a_terminal_run_refuses_another_segment_even_without_the_resume_flag(tmp_path):
    """``completed`` is terminal, not "terminal as long as you asked to resume". Opening an
    existing run by id without ``resume=True`` must refuse it too — otherwise a caller that
    passes the two apart could append a segment to a published (or pruned) run."""
    from noctis.reporting.run_store import RunCompletedError

    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    _prune(runs, run_dir.name, clock)

    with pytest.raises(RunCompletedError):
        _open(runs, clock, run_id=run_dir.name, resume=False)


def test_a_dry_run_reports_what_it_would_remove_and_leaves_every_byte_on_disk(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, sizes = _completed_with_state(runs, clock)
    before = (run_dir / "run.json").read_bytes()
    index_before = (runs / "index.json").read_bytes()

    outcome = _prune(runs, run_dir.name, clock, dry_run=True)

    assert outcome.dry_run is True
    assert sorted(outcome.removed) == ["reports", "state", "strategies"]
    assert outcome.freed_bytes == sizes["state"] + sizes["strategies"] + sizes["reports"]
    for name in ("state", "strategies", "reports"):
        assert _dir_bytes(run_dir / name) == sizes[name]  # every byte still there
    assert (run_dir / "run.json").read_bytes() == before  # nothing written, not even the marker
    assert (runs / "index.json").read_bytes() == index_before


def test_pruning_twice_removes_nothing_the_second_time(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)
    _prune(runs, run_dir.name, clock)

    again = _prune(runs, run_dir.name, clock)

    assert again.removed == ()
    assert again.freed_bytes == 0
    assert _on_disk(run_dir)["run"]["state_pruned"] is True


def test_pruning_never_reaches_outside_the_run_it_addressed(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    neighbour = _completed_run(runs, clock)
    clock.advance(HOUR)
    target = _completed_run(runs, clock)
    _fill(neighbour)
    sizes = _fill(target)

    _prune(runs, target.name, clock)

    assert _dir_bytes(neighbour / "state") == sizes["state"]
    assert (neighbour / "strategies" / "__tmp" / "draft.py").exists()
    assert (neighbour / "reports").is_dir()


def test_a_symlinked_state_directory_is_left_alone_and_its_target_survives(tmp_path):
    """Retention removes directories *in* the run tree. It never follows a link out of one, so a
    workspace someone symlinked elsewhere cannot be deleted through a run."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir = _completed_run(runs, clock)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "experiments").mkdir(parents=True)
    (elsewhere / "experiments" / "momentum.jsonl").write_text("precious")
    (run_dir / "state").symlink_to(elsewhere, target_is_directory=True)
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "2026-07-27.md").write_text("r" * 40)

    outcome = _prune(runs, run_dir.name, clock)

    assert outcome.removed == ("reports",)  # the link is not one of ours to remove
    assert (elsewhere / "experiments" / "momentum.jsonl").read_text() == "precious"
    assert (run_dir / "state").is_symlink()


def test_a_file_named_like_a_pruned_directory_is_never_removed(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir = _completed_run(runs, clock)
    (run_dir / "state").write_text("not a directory")

    outcome = _prune(runs, run_dir.name, clock)

    assert outcome.removed == ()
    assert (run_dir / "state").read_text() == "not a directory"


def test_a_directory_that_is_not_a_run_cannot_be_pruned_through_the_path_form(tmp_path):
    """The path address form honours wherever it points, so the record is the gate: no ``run.json``
    saying ``completed`` means nothing is deleted, whatever the operator typed."""
    from noctis.reporting.run_store import RunNotFoundError

    precious = tmp_path / "home"
    (precious / "state").mkdir(parents=True)
    (precious / "state" / "keep.txt").write_text("mine")

    with pytest.raises(RunNotFoundError):
        _prune(tmp_path / "runs", str(precious), FakeClock())

    assert (precious / "state" / "keep.txt").read_text() == "mine"


def test_pruning_an_unknown_run_is_a_clean_lookup_failure(tmp_path):
    from noctis.reporting.run_store import RunNotFoundError

    with pytest.raises(RunNotFoundError):
        _prune(tmp_path / "runs", "20260101T000000Z-nope00", FakeClock())


def test_a_pruned_run_still_lists_in_the_index(tmp_path):
    from noctis.reporting.run_store import index_entry, rebuild_index

    runs = tmp_path / "runs"
    clock = FakeClock()
    run_dir, _ = _completed_with_state(runs, clock)

    _prune(runs, run_dir.name, clock)

    entry = index_entry(run_dir)
    assert entry["readable"] is True
    assert entry["status"] == "completed"
    assert entry["cumulative_runtime_s"] == HOUR
    assert [e["run_id"] for e in rebuild_index(runs)["runs"]] == [run_dir.name]


def test_opening_and_closing_a_run_never_marks_it_pruned(tmp_path):
    """The default is to keep everything: nothing in the ordinary write path sets the marker."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    clock.advance(HOUR)
    store.close(reason="stop_requested")

    assert _on_disk(store.run_dir)["run"]["state_pruned"] is False


# ── the CLI: `noctis run-prune <address> [--dry-run]` ──────────────────────────────────────


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
    return json.loads((_runs_dir(tmp_path) / run_id / "run.json").read_text())


def _cli_run(tmp_path: Path, cfg: str, *, finish: bool = True) -> tuple[str, dict[str, int]]:
    """One real ``noctis run``, given the heavy tree a working night leaves, optionally sealed."""
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    run_id = _run_ids(tmp_path)[-1]
    sizes = _fill(_runs_dir(tmp_path) / run_id)
    if finish:
        assert (
            runner.invoke(app, ["run", "--config", cfg, "--resume", run_id, "--finish"]).exit_code
            == 0
        )
    return run_id, sizes


def test_noctis_run_prune_removes_a_completed_runs_state_and_says_what_it_freed(tmp_path):
    cfg = _config(tmp_path)
    run_id, sizes = _cli_run(tmp_path, cfg)

    result = runner.invoke(app, ["run-prune", run_id, "--config", cfg])

    assert result.exit_code == 0, result.output
    run_dir = _runs_dir(tmp_path) / run_id
    assert not (run_dir / "state").exists()
    assert not (run_dir / "strategies").exists()
    assert not (run_dir / "reports").exists()
    assert (run_dir / "run.json").is_file() and (_runs_dir(tmp_path) / "index.json").is_file()
    assert _cli_record(tmp_path, run_id)["run"]["state_pruned"] is True
    freed = sizes["state"] + sizes["strategies"] + sizes["reports"]
    assert str(freed) in result.output or f"{freed / 1024:.1f}" in result.output
    assert run_id in result.output


def test_noctis_run_prune_dry_run_reports_what_it_would_remove_and_removes_nothing(tmp_path):
    cfg = _config(tmp_path)
    run_id, sizes = _cli_run(tmp_path, cfg)
    run_dir = _runs_dir(tmp_path) / run_id
    before = (run_dir / "run.json").read_bytes()

    result = runner.invoke(app, ["run-prune", run_id, "--config", cfg, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "state" in result.output and "strategies" in result.output
    for name in ("state", "strategies", "reports"):
        assert _dir_bytes(run_dir / name) == sizes[name]
    assert (run_dir / "run.json").read_bytes() == before
    assert _cli_record(tmp_path, run_id)["run"]["state_pruned"] is False


def test_noctis_run_prune_refuses_a_run_that_could_still_be_resumed(tmp_path):
    cfg = _config(tmp_path)
    run_id, sizes = _cli_run(tmp_path, cfg, finish=False)

    result = runner.invoke(app, ["run-prune", run_id, "--config", cfg])

    assert result.exit_code != 0
    assert "resum" in result.output
    run_dir = _runs_dir(tmp_path) / run_id
    assert _dir_bytes(run_dir / "state") == sizes["state"]
    # …and the run it protected really does still resume.
    assert runner.invoke(app, ["run", "--config", cfg, "--resume", run_id]).exit_code == 0


def test_nothing_is_pruned_unless_an_operator_asks(tmp_path):
    """Retention is opt-in: the default keeps everything. Running, sealing and listing runs never
    removes a byte — only the verb does."""
    cfg = _config(tmp_path)
    run_id, sizes = _cli_run(tmp_path, cfg)

    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0  # a second night
    assert runner.invoke(app, ["runs", "--config", cfg, "--all"]).exit_code == 0
    assert runner.invoke(app, ["run-record", run_id, "--config", cfg]).exit_code == 0

    run_dir = _runs_dir(tmp_path) / run_id
    assert {name: _dir_bytes(run_dir / name) for name in sizes} == sizes
    assert _cli_record(tmp_path, run_id)["run"]["state_pruned"] is False


def test_a_pruned_run_still_lists_and_still_prints(tmp_path):
    cfg = _config(tmp_path)
    run_id, _ = _cli_run(tmp_path, cfg)
    assert runner.invoke(app, ["run-prune", run_id, "--config", cfg]).exit_code == 0

    listed = runner.invoke(app, ["runs", "--config", cfg, "--all"])
    printed = runner.invoke(app, ["run-record", run_id, "--config", cfg])

    assert listed.exit_code == 0, listed.output
    assert run_id in listed.output and "completed" in listed.output
    assert printed.exit_code == 0, printed.output
    record = json.loads(printed.output)
    assert record["run"]["run_id"] == run_id
    assert record["run"]["state_pruned"] is True
    assert schema.validate(record) == []


def test_a_pruned_run_refuses_the_next_resume_from_the_cli(tmp_path):
    cfg = _config(tmp_path)
    run_id, _ = _cli_run(tmp_path, cfg)
    assert runner.invoke(app, ["run-prune", run_id, "--config", cfg]).exit_code == 0

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert result.exit_code != 0
    assert "completed" in result.output
    assert len(_run_ids(tmp_path)) == 1  # refused, never quietly minted a new run


def test_noctis_run_prune_of_an_unknown_address_removes_nothing(tmp_path):
    cfg = _config(tmp_path)
    run_id, sizes = _cli_run(tmp_path, cfg)

    result = runner.invoke(app, ["run-prune", "20260101T000000Z-nope00", "--config", cfg])

    assert result.exit_code != 0
    assert _dir_bytes(_runs_dir(tmp_path) / run_id / "state") == sizes["state"]
