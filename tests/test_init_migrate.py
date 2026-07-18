"""`noctis init` (idempotent scaffold) and `noctis migrate` (one-shot legacy move)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from noctis.cli import app

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


# ── migrate ───────────────────────────────────────────────────────────────────────────────
def test_migrate_moves_all_six_legacy_artifacts(tmp_path, monkeypatch):
    from noctis.bootstrap import detect_legacy_layout
    from noctis.config import load_settings

    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _legacy_layout(root)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    ws = tmp_path / "workspace"
    assert (ws / "state" / "marker.txt").is_file()
    assert (ws / "data_lake" / "marker.txt").is_file()
    assert (ws / "reports" / "marker.txt").is_file()
    assert "legacy memory" in (ws / "memory" / "MEMORY.md").read_text()
    assert (ws / "strategies" / "__tmp" / "marker.txt").is_file()
    assert (ws / "strategies" / "champions" / "marker.txt").is_file()
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
    assert not (tmp_path / "workspace" / "state").exists()


def test_migrate_refuses_with_a_list_when_both_copies_exist(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    _legacy_layout(root)
    (tmp_path / "workspace" / "state").mkdir(parents=True)  # a workspace copy already exists
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 2
    assert "state" in result.output
    assert (root / "state" / "marker.txt").is_file()  # NOTHING moved, not even clean pairs
    assert not (tmp_path / "workspace" / "reports").exists()


def test_migrate_skips_a_knob_pinned_to_the_legacy_path(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\nstate_dir: state/\n")
    _legacy_layout(root)
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert (root / "state" / "marker.txt").is_file()  # pinned → left in place, with a note
    assert "state" in result.output
    assert (tmp_path / "workspace" / "reports" / "marker.txt").is_file()  # others still move


def test_migrate_with_nothing_legacy_is_a_polite_no_op(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text("mode: paper\n")
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert "Nothing to migrate" in result.output
