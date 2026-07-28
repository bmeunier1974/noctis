"""`noctis init` (idempotent scaffold) and `noctis migrate` (one-shot legacy move).

``migrate`` moves two generations of legacy layout into where the engine now reads:

* the **pre-workspace** artifacts beside ``config.yaml`` (``state/``, ``data_lake/``, ``reports/``,
  ``MEMORY.md``, the two strategy tiers), and
* the **pre-run-scoped** workspace artifacts (``workspace/state/`` and friends), adopted into the
  reserved ``legacy`` run so an existing operator's champions, account and reports survive and
  become their first resumable run (story #131).

Both hops are the same idea and the same one instruction to the operator, so they share one plan,
one ``--dry-run``, one conflict refusal, and one command.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from noctis.cli import app
from noctis.config.settings import DEFAULT_RUN_ID

runner = CliRunner()


def _project(tmp_path, monkeypatch, *, templates: bool = True) -> Path:
    """A project root the CLI runs in (chdir'd; config resolves here)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOCTIS_CONFIG", str(tmp_path / "config.yaml"))
    if templates:
        (tmp_path / "config.example.yaml").write_text("mode: paper\n")
        (tmp_path / ".env.example").write_text("ALLOW_LIVE=\n")
        (tmp_path / "mandate").mkdir()
        (tmp_path / "mandate" / "MANDATE.md.example").write_text("# my mandate\n")
    return tmp_path


def _legacy_layout(root: Path) -> None:
    """The six legacy artifacts, each with a marker file."""
    for d in ("state", "data_lake", "reports", "strategies/__tmp", "strategies/champions"):
        (root / d).mkdir(parents=True)
        (root / d / "marker.txt").write_text(d)
    (root / "MEMORY.md").write_text("# MEMORY\n\nlegacy memory\n")


# ── init ──────────────────────────────────────────────────────────────────────────────────
def test_init_scaffolds_the_three_local_files_and_the_workspace(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (root / "config.yaml").read_text() == "mode: paper\n"
    assert (root / ".env").read_text() == "ALLOW_LIVE=\n"
    assert (root / "mandate" / "MANDATE.md").read_text() == "# my mandate\n"
    assert (tmp_path / "workspace").is_dir()  # conftest pins NOCTIS_WORKSPACE here


def test_init_is_idempotent_and_never_overwrites_edits(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    assert runner.invoke(app, ["init"]).exit_code == 0
    (root / "config.yaml").write_text("mode: paper\nchampion_count: 5\n")  # the user's edit
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (root / "config.yaml").read_text() == "mode: paper\nchampion_count: 5\n"
    assert "kept" in result.output


def test_init_survives_a_missing_template(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, templates=False)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "workspace").is_dir()


# ── migrate: the pre-workspace hop ────────────────────────────────────────────────────────
def _run_dir(tmp_path: Path) -> Path:
    """Where the run-scoped artifacts now live: the reserved run under the workspace."""
    return tmp_path / "workspace" / "runs" / DEFAULT_RUN_ID


def test_migrate_moves_all_six_legacy_artifacts(tmp_path, monkeypatch):
    """The pre-workspace artifacts land where the engine reads them *now* — the run-scoped
    locations for the five that a run owns, and the workspace-level lake for the one it shares."""
    from noctis.bootstrap import detect_legacy_layout
    from noctis.config import load_settings

    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _legacy_layout(root)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    run = _run_dir(tmp_path)
    assert (run / "state" / "marker.txt").is_file()
    assert (tmp_path / "workspace" / "data_lake" / "marker.txt").is_file()  # shared, not run-scoped
    assert (run / "reports" / "marker.txt").is_file()
    assert "legacy memory" in (run / "memory" / "MEMORY.md").read_text()
    assert (run / "strategies" / "__tmp" / "marker.txt").is_file()
    assert (run / "strategies" / "champions" / "marker.txt").is_file()
    for gone in ("state", "data_lake", "reports", "MEMORY.md"):
        assert not (root / gone).exists()
    # config.yaml never moves, and the guard now admits every command.
    assert (root / "config.yaml").is_file()
    assert detect_legacy_layout(load_settings()) == []


def test_migrate_dry_run_lists_the_plan_and_mutates_nothing(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _legacy_layout(root)
    result = runner.invoke(app, ["migrate", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert result.output.count("would move") == 6
    assert (root / "state" / "marker.txt").is_file()  # untouched
    assert not _run_dir(tmp_path).exists()


def test_migrate_refuses_with_a_list_when_two_copies_target_one_location(tmp_path, monkeypatch):
    """Two legacy generations of the same artifact cannot both be adopted into one location.

    (Before run-scoped state this read "the workspace copy already exists, so there is nothing to
    do". It no longer can: ``workspace/state/`` is itself un-adopted now, so both copies want the
    reserved run's ``state/`` and only a human can say which one is the history.)
    """
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _legacy_layout(root)
    (tmp_path / "workspace" / "state").mkdir(parents=True)  # a workspace-era copy as well
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 2
    assert "state" in result.output
    assert (root / "state" / "marker.txt").is_file()  # NOTHING moved, not even clean pairs
    assert not (_run_dir(tmp_path) / "reports").exists()


def test_migrate_skips_a_knob_pinned_to_the_legacy_path(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\nstate_dir: state/\n")
    _legacy_layout(root)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert (root / "state" / "marker.txt").is_file()  # pinned → left in place, with a note
    assert "state" in result.output
    assert (_run_dir(tmp_path) / "reports" / "marker.txt").is_file()  # others still move


# ── migrate: adopting pre-run-scoped workspace state into the reserved `legacy` run ────────
def _workspace_state(tmp_path: Path) -> Path:
    """A pre-run-scoped workspace: champions, an account, a report, memory and both tiers."""
    ws = tmp_path / "workspace"
    (ws / "state").mkdir(parents=True)
    (ws / "state" / "champions.json").write_text(json.dumps({"champions": [], "history": []}))
    (ws / "state" / "paper_account.json").write_text(json.dumps({"equity": 101_234.5}))
    (ws / "reports").mkdir(parents=True)
    (ws / "reports" / "2026-07-01.md").write_text("# Close-of-day report — 2026-07-01\n")
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "MEMORY.md").write_text("# MEMORY\n\n- a lesson learned before run scoping\n")
    for tier in ("__tmp", "champions"):
        (ws / "strategies" / tier).mkdir(parents=True)
        (ws / "strategies" / tier / f"{tier}_marker.txt").write_text(tier)
    return ws


def test_migrate_adopts_workspace_state_into_the_reserved_legacy_run(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    ws = _workspace_state(tmp_path)

    result = runner.invoke(app, ["migrate"])

    assert result.exit_code == 0, result.output
    run = _run_dir(tmp_path)
    assert json.loads((run / "state" / "paper_account.json").read_text())["equity"] == 101_234.5
    assert (run / "state" / "champions.json").is_file()
    assert (run / "reports" / "2026-07-01.md").is_file()
    assert "before run scoping" in (run / "memory" / "MEMORY.md").read_text()
    assert (run / "strategies" / "__tmp" / "__tmp_marker.txt").is_file()
    assert (run / "strategies" / "champions" / "champions_marker.txt").is_file()
    for gone in ("state", "reports"):
        assert not (ws / gone).exists()


def test_the_adopted_run_is_a_real_run_with_its_own_record(tmp_path, monkeypatch):
    """It needs a ``run.json`` like any other run: listable today, resumable later."""
    from noctis.reporting import schema

    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)

    assert runner.invoke(app, ["migrate"]).exit_code == 0

    record = json.loads((_run_dir(tmp_path) / "run.json").read_text())
    assert schema.validate(record) == []
    assert record["run"]["run_id"] == DEFAULT_RUN_ID
    assert record["run"]["status"] == "stopped"  # resumable, like any cleanly-stopped run
    assert any("adopt" in event["text"] for event in record["events"])

    listed = runner.invoke(app, ["runs"])
    assert listed.exit_code == 0, listed.output
    assert DEFAULT_RUN_ID in listed.output


def test_the_reserved_run_is_addressed_like_any_other(tmp_path, monkeypatch):
    """Its id is a fixed string rather than a minted one; nothing else about it is special."""
    from noctis.config import load_settings
    from noctis.reporting.run_store import resolve_run_dir

    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)
    assert runner.invoke(app, ["migrate"]).exit_code == 0

    settings = load_settings()
    assert resolve_run_dir(settings.runs_dir, DEFAULT_RUN_ID) == _run_dir(tmp_path)

    printed = runner.invoke(app, ["run-record", DEFAULT_RUN_ID])
    assert printed.exit_code == 0, printed.output
    assert json.loads(printed.output)["run"]["run_id"] == DEFAULT_RUN_ID


def test_migrate_dry_run_reports_the_adoption_without_performing_it(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    ws = _workspace_state(tmp_path)

    result = runner.invoke(app, ["migrate", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.output.count("would move") == 5  # state, reports, memory, and the two tiers
    assert DEFAULT_RUN_ID in result.output
    assert (ws / "state" / "champions.json").is_file()  # untouched
    assert not _run_dir(tmp_path).exists()


def test_migrating_twice_neither_duplicates_nor_destroys_state(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)

    assert runner.invoke(app, ["migrate"]).exit_code == 0
    run = _run_dir(tmp_path)
    record_before = (run / "run.json").read_text()
    account_before = (run / "state" / "paper_account.json").read_text()

    second = runner.invoke(app, ["migrate"])

    assert second.exit_code == 0, second.output
    assert "Nothing to migrate" in second.output
    assert (run / "state" / "paper_account.json").read_text() == account_before
    assert (run / "run.json").read_text() == record_before
    assert not (run / "state" / "state").exists()  # never nested inside itself


def test_migrate_refuses_when_the_run_already_holds_that_artifact(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)
    (_run_dir(tmp_path) / "state").mkdir(parents=True)  # the run already has its own state

    result = runner.invoke(app, ["migrate"])

    assert result.exit_code == 2
    assert "state" in result.output
    assert (tmp_path / "workspace" / "state" / "champions.json").is_file()  # nothing moved


# ── the startup guard: un-adopted workspace state warns, it never refuses ─────────────────
def test_run_warns_on_unadopted_workspace_state_and_still_starts(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n")
    _workspace_state(tmp_path)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    assert "noctis migrate" in result.output
    assert "workspace" in result.output


def test_status_warns_on_unadopted_workspace_state_rather_than_refusing(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "noctis migrate" in result.output
    assert "mode (resolved):" in result.output


def test_no_warning_once_the_state_has_been_adopted(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _workspace_state(tmp_path)
    assert runner.invoke(app, ["migrate"]).exit_code == 0

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "noctis migrate" not in result.output


def test_migrate_with_nothing_legacy_is_a_polite_no_op(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert "Nothing to migrate" in result.output
