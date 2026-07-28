"""The engine-change resume policy — ``--allow-engine-upgrade`` (story #135, epic #126).

A run resumed after ``git pull`` may find a different engine, and the policy splits on **who
changed: the judge or the searcher**. Arbiter drift refuses (champions crowned under two sets of
gates must never accumulate inside one experiment); searcher drift warns, records and proceeds
(improving how candidates are found must not invalidate a run whose arbiter held still); no drift
is silent. ``--allow-engine-upgrade`` overrides the refusal — deliberately, on the record, and
never invisibly.

Two techniques, both external-behaviour only. The pure policy is exercised the way #127/#128
exercise the fingerprint: build a miniature repo in ``tmp_path``, fingerprint it, edit one file,
fingerprint it again. The CLI is exercised by editing the **record's** frozen fingerprint, which
is the same disagreement seen from the run's side: the run says the engine was X, this checkout
computes Y.
"""

from __future__ import annotations

import ast
import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.observability import engine_change as policy
from noctis.observability.engine_change import (
    ACCEPTED_BY,
    EngineChangeError,
    assert_arbiter_held,
    engine_change,
    engine_notes,
    upgrade_entry,
)
from noctis.observability.engine_id import COMPONENT_PATHS, fingerprint
from noctis.reporting import schema
from noctis.reporting.run_store import RUN_RECORD_NAME, open_run

runner = CliRunner()

GATES_FILE = "src/noctis/champions/promotion.py"
BACKTEST_FILE = "src/noctis/backtest/pipeline.py"
PROMPTS_FILE = "src/noctis/research/prompt.py"


# ── the pure policy, over a miniature repo ─────────────────────────────────────────────────


def _build_tree(root: Path) -> Path:
    """A miniature repo: every path the component map names, content unique per path."""
    for paths in COMPONENT_PATHS.values():
        for rel in paths:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {rel}\nbody of {rel}\n")
    return root


def _edit(root: Path, rel: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + "# a behavioural change\n")


def _record_of(root: Path, *, epoch: int = 1) -> dict:
    """A run record whose ``engine`` section froze the fingerprint of ``root``."""
    fp = fingerprint(root)
    return {
        "run": {"run_id": "20260727T142233Z-a1b2c3"},
        "segments": [],
        "engine": {
            "engine_version": fp.engine_version,
            "engine_epoch": epoch,
            "noctis_version": "0.1.0",
            "fingerprint": fp.digests(),
            "comparable_key": "1|frozen|frozen|sharpe",
            "mixed_engine": False,
            "engine_changes": [],
        },
    }


def test_the_frozen_fingerprint_is_compared_with_the_current_one_component_by_component(tmp_path):
    """Not one opaque hash: the verdict names which component moved, and both its digests."""
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    _edit(root, PROMPTS_FILE)

    change = engine_change(frozen, fingerprint(root))

    assert [component.component for component in change.components] == ["prompts"]
    (moved,) = change.components
    assert moved.frozen == frozen["engine"]["fingerprint"]["prompts"]
    assert moved.current == fingerprint(root).digest("prompts")
    assert moved.frozen != moved.current
    assert moved.tier == "searcher"


def test_an_unchanged_engine_is_no_change_at_all(tmp_path):
    root = _build_tree(tmp_path)

    change = engine_change(_record_of(root), fingerprint(root))

    assert not change
    assert change.components == ()
    assert engine_notes(change) == ()


def test_gates_drift_refuses_the_resume_with_a_message_naming_the_component(tmp_path):
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    _edit(root, GATES_FILE)

    change = engine_change(frozen, fingerprint(root))
    with pytest.raises(EngineChangeError) as excinfo:
        assert_arbiter_held(change, run_id="20260727T142233Z-a1b2c3")

    assert "gates" in str(excinfo.value)
    assert "20260727T142233Z-a1b2c3" in str(excinfo.value)
    assert ACCEPTED_BY in str(excinfo.value)


def test_backtest_drift_refuses_the_resume_with_a_message_naming_the_component(tmp_path):
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    _edit(root, BACKTEST_FILE)

    change = engine_change(frozen, fingerprint(root))
    with pytest.raises(EngineChangeError) as excinfo:
        assert_arbiter_held(change, run_id="20260727T142233Z-a1b2c3")

    assert "backtest" in str(excinfo.value)
    assert [component.tier for component in change.components] == ["arbiter"]


def test_searcher_drift_never_refuses_and_is_recorded_naming_the_component_and_its_files(tmp_path):
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    _edit(root, PROMPTS_FILE)

    change = engine_change(frozen, fingerprint(root))
    assert_arbiter_held(change, run_id="a-run")  # no refusal: the arbiter held still
    (note,) = engine_notes(change)

    assert "prompts" in note
    for rel in COMPONENT_PATHS["prompts"]:
        assert rel in note
    assert upgrade_entry(change, at="2026-07-28T01:10:04.002Z", segment=1) is None


def test_allow_engine_upgrade_overrides_the_refusal_and_builds_the_recorded_change(tmp_path):
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    _edit(root, GATES_FILE)
    _edit(root, PROMPTS_FILE)

    change = engine_change(frozen, fingerprint(root))
    assert_arbiter_held(change, run_id="a-run", upgrading=True)  # the escape hatch
    entry = upgrade_entry(change, at="2026-07-28T01:10:04.002Z", segment=1)

    assert entry is not None
    assert entry["from_epoch"] == 1 and entry["to_epoch"] == 2
    assert entry["segment"] == 1
    assert entry["accepted_by"] == ACCEPTED_BY
    # Every component that moved is named, with its tier and both digests.
    assert [component["component"] for component in entry["components"]] == ["gates", "prompts"]
    assert [component["tier"] for component in entry["components"]] == ["arbiter", "searcher"]
    assert any(ACCEPTED_BY in note for note in engine_notes(change, upgrading=True))


def test_a_component_only_one_side_knows_about_counts_as_drift_and_two_nulls_do_not(tmp_path):
    """The same missing-input rule the fingerprint itself takes: nothing known to have moved is
    not drift, but a component that appeared or vanished is."""
    root = _build_tree(tmp_path)
    frozen = _record_of(root)
    frozen["engine"]["fingerprint"].pop("seeds")
    frozen["engine"]["fingerprint"]["memory_seed"] = None
    (root / "MEMORY.seed.md").unlink()

    change = engine_change(frozen, fingerprint(root))

    assert [component.component for component in change.components] == ["seeds"]


def test_a_record_with_no_readable_engine_section_reports_no_drift(tmp_path):
    """A run adopted from pre-record history froze no engine — there is no side to compare, and
    stranding exactly the history the adoption path preserves would be the wrong answer."""
    root = _build_tree(tmp_path)

    assert not engine_change({}, fingerprint(root))
    assert not engine_change({"engine": {"fingerprint": None}}, fingerprint(root))


# ── the record: an engine frozen at creation is what there is to compare against ───────────


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self) -> None:
        self.now = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _open(runs_dir: Path, clock: FakeClock, tree: Path, **kwargs):
    return open_run(
        runs_dir,
        clock=clock,
        argv=["run", "-v"],
        election_metric="sharpe",
        engine_root=tree,
        **kwargs,
    )


def test_the_run_level_engine_is_frozen_at_creation_and_each_segment_carries_its_own(tmp_path):
    """Without this there is nothing to compare a resume against: a record that restamped the
    current engine on every write could only ever say the engine equals itself."""
    tree = _build_tree(tmp_path / "tree")
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, tree)
    first.close(reason="time_limit")
    frozen = _record(first.run_dir)["engine"]

    _edit(tree, PROMPTS_FILE)
    clock.advance(3600)
    second = _open(runs, clock, tree, run_id=first.run_id, resume=True)
    second.close(reason="stopped")

    record = _record(first.run_dir)
    assert record["engine"]["fingerprint"] == frozen["fingerprint"]
    assert record["engine"]["comparable_key"] == frozen["comparable_key"]
    assert record["engine"]["engine_epoch"] == 1
    assert record["engine"]["engine_changes"] == []
    # …while the segment records the engine that actually produced it.
    assert record["segments"][0]["engine_fingerprint"] == frozen["fingerprint"]
    assert record["segments"][1]["engine_fingerprint"] != frozen["fingerprint"]
    assert record["engine"]["mixed_engine"] is True


def test_an_accepted_upgrade_re_freezes_the_engine_bumping_the_epoch_and_keeping_the_history(
    tmp_path,
):
    tree = _build_tree(tmp_path / "tree")
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, tree)
    first.close(reason="time_limit")
    frozen = _record(first.run_dir)["engine"]

    _edit(tree, GATES_FILE)
    clock.advance(3600)
    change = engine_change(_record(first.run_dir), fingerprint(tree))
    entry = upgrade_entry(change, at="2026-07-27T15:22:33.418Z", segment=1)
    second = _open(runs, clock, tree, run_id=first.run_id, resume=True, engine_upgrade=entry)
    second.close(reason="stopped")

    engine = _record(first.run_dir)["engine"]
    assert engine["engine_epoch"] == 2
    assert engine["fingerprint"] == fingerprint(tree).digests()  # re-frozen on the new engine
    assert engine["comparable_key"] != frozen["comparable_key"]  # a new bucket, honestly
    assert engine["engine_changes"] == [entry]
    assert engine["mixed_engine"] is True
    assert schema.validate(_record(first.run_dir)) == []


# ── the CLI: refusing, upgrading, warning, and staying silent ──────────────────────────────


def _config(tmp_path: Path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}")
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _started(tmp_path: Path) -> tuple[str, Path]:
    """One finished run, and the run dir its record sits in."""
    cfg = _config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg])
    assert result.exit_code == 0, result.output
    (run_dir,) = [p for p in _runs_dir(tmp_path).iterdir() if p.is_dir()]
    return cfg, run_dir


def _move_component(run_dir: Path, component: str) -> str:
    """Rewrite the run's frozen digest for one component — the engine moved under it."""
    record = _record(run_dir)
    frozen = record["engine"]["fingerprint"][component]
    record["engine"]["fingerprint"][component] = "0" * 16
    (run_dir / RUN_RECORD_NAME).write_text(json.dumps(record))
    return frozen


def test_resuming_across_gates_drift_refuses_and_opens_no_segment(tmp_path):
    cfg, run_dir = _started(tmp_path)
    _move_component(run_dir, "gates")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 1
    assert "gates" in result.output
    assert "--allow-engine-upgrade" in result.output
    assert len(_record(run_dir)["segments"]) == 1  # the refusal landed before a segment opened


def test_resuming_across_backtest_drift_refuses_naming_the_component(tmp_path):
    cfg, run_dir = _started(tmp_path)
    _move_component(run_dir, "backtest")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 1
    assert "backtest" in result.output


def test_allow_engine_upgrade_proceeds_bumps_the_epoch_records_it_and_flags_mixed_engine(tmp_path):
    cfg, run_dir = _started(tmp_path)
    before = _move_component(run_dir, "gates")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--allow-engine-upgrade"]
    )

    assert result.exit_code == 0, result.output
    engine = _record(run_dir)["engine"]
    assert engine["engine_epoch"] == 2
    (change,) = engine["engine_changes"]
    assert [component["component"] for component in change["components"]] == ["gates"]
    assert change["components"][0]["from"] == "0" * 16
    assert change["components"][0]["to"] == before
    assert change["segment"] == 1
    assert change["accepted_by"] == "--allow-engine-upgrade"
    assert engine["mixed_engine"] is True
    assert "engine_epoch" in result.output and "gates" in result.output


def test_searcher_drift_warns_records_an_event_and_proceeds(tmp_path):
    cfg, run_dir = _started(tmp_path)
    _move_component(run_dir, "prompts")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    record = _record(run_dir)
    assert len(record["segments"]) == 2  # it proceeded
    (event,) = [event for event in record["events"] if "prompts" in event["text"]]
    assert event["segment"] == 1
    for rel in COMPONENT_PATHS["prompts"]:
        assert rel in event["text"]
    assert "prompts" in result.output
    # …and the searcher moving never bumps the epoch: the arbiter is what an epoch is about.
    assert record["engine"]["engine_epoch"] == 1
    assert record["engine"]["engine_changes"] == []


def test_a_resume_with_no_engine_drift_says_nothing_and_records_nothing(tmp_path):
    """A policy that always logs something is a policy operators learn to ignore."""
    cfg, run_dir = _started(tmp_path)

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    record = _record(run_dir)
    assert [event["text"] for event in record["events"]] == []
    assert record["engine"]["mixed_engine"] is False
    assert record["engine"]["engine_epoch"] == 1
    assert "engine_epoch" not in result.output
    assert "drift" not in result.output.lower()


def test_a_run_resumed_after_searcher_drift_keeps_its_comparable_key(tmp_path):
    """The arbiter held still and the metric did not move, so the run stays in its bucket —
    that is the whole point of splitting the fingerprint by component."""
    cfg, run_dir = _started(tmp_path)
    before = _record(run_dir)["engine"]["comparable_key"]
    _move_component(run_dir, "seeds")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    assert _record(run_dir)["engine"]["comparable_key"] == before


def test_mixed_engine_is_visible_in_the_record_and_in_noctis_runs(tmp_path):
    cfg, run_dir = _started(tmp_path)
    _move_component(run_dir, "gates")
    upgraded = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--allow-engine-upgrade"]
    )
    assert upgraded.exit_code == 0, upgraded.output

    listed = runner.invoke(app, ["runs", "--all", "--config", cfg])

    assert listed.exit_code == 0, listed.output
    assert _record(run_dir)["engine"]["mixed_engine"] is True
    index = json.loads((_runs_dir(tmp_path) / "index.json").read_text())
    assert [entry["mixed_engine"] for entry in index["runs"]] == [True]
    assert "mixed engine" in listed.output


def test_allow_engine_upgrade_without_resume_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path), "--allow-engine-upgrade"])

    assert result.exit_code == 1
    assert "--resume" in result.output


def test_allow_engine_upgrade_with_nothing_to_upgrade_is_a_documented_no_op(tmp_path):
    """No arbiter drift, no bump: an epoch that moved for nothing would mark the run as
    engine-changed forever."""
    cfg, run_dir = _started(tmp_path)

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--allow-engine-upgrade"]
    )

    assert result.exit_code == 0, result.output
    engine = _record(run_dir)["engine"]
    assert engine["engine_epoch"] == 1
    assert engine["engine_changes"] == []
    assert engine["mixed_engine"] is False


# ── the policy is never a gate ─────────────────────────────────────────────────────────────


def test_the_policy_module_reads_no_config_no_clock_and_no_gate():
    """Evidence and policy, never a gate: this decides whether a run may *continue*, and it
    reaches nothing that decides what passes, no clock and no configuration."""
    allowed = {"__future__", "collections", "dataclasses", "typing", "noctis"}
    tree = ast.parse(Path(policy.__file__).read_text())
    imported: set[str] = set()
    inner: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            if node.module.startswith("noctis"):
                inner.add(node.module)

    assert imported <= allowed, imported - allowed
    assert inner == {"noctis.observability.engine_id"}, inner
