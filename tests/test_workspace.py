"""Workspace root: path derivation + the legacy-layout guard (epic #39).

The behaviors under test are external: where the engine is told to write (one
``workspace_dir`` root; per-artifact knobs derive from it unless explicitly set) and
whether commands refuse to run beside un-migrated legacy data.

Since story #131 the derivation has two links: the workspace owns the run tree and the shared
data lake, and a **run dir** under it owns the four per-run artifacts (state, reports, qa,
memory). This suite still asserts the workspace end of that chain — ``tests/test_run_scoped_state``
asserts the run end — so every expectation here goes through the reserved default run.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from noctis.config import load_settings


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body))
    return path


class TestWorkspaceDerivation:
    @pytest.fixture(autouse=True)
    def _no_pinned_workspace(self, monkeypatch):
        # conftest pins NOCTIS_WORKSPACE to tmp_path (ambient-write isolation); these tests
        # assert the *default* derivation, so they run with the variable genuinely unset.
        monkeypatch.delenv("NOCTIS_WORKSPACE", raising=False)

    def test_all_paths_derive_from_the_default_workspace(self, tmp_path):
        settings = load_settings(config_path=tmp_path / "missing.yaml")
        assert Path(settings.workspace_dir) == Path("workspace")
        assert Path(settings.runs_dir) == Path("workspace/runs")
        assert Path(settings.run_dir) == Path("workspace/runs/legacy")
        assert Path(settings.state_dir) == Path("workspace/runs/legacy/state")
        assert Path(settings.reports_dir) == Path("workspace/runs/legacy/reports")
        assert Path(settings.memory_path) == Path("workspace/runs/legacy/memory/MEMORY.md")
        assert Path(settings.qa_dir) == Path("workspace/runs/legacy/qa")
        # The lake is shared by every run, so it hangs off the workspace, not off a run.
        assert Path(settings.data.lake_dir) == Path("workspace/data_lake")

    def test_env_noctis_workspace_rebases_every_derived_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NOCTIS_WORKSPACE", str(tmp_path / "out"))
        settings = load_settings(config_path=tmp_path / "missing.yaml")
        out = tmp_path / "out"
        run = out / "runs" / "legacy"
        assert Path(settings.workspace_dir) == out
        assert Path(settings.state_dir) == run / "state"
        assert Path(settings.reports_dir) == run / "reports"
        assert Path(settings.memory_path) == run / "memory" / "MEMORY.md"
        assert Path(settings.qa_dir) == run / "qa"
        assert Path(settings.data.lake_dir) == out / "data_lake"

    def test_yaml_workspace_dir_rebases_every_derived_path(self, tmp_path):
        cfg = _write_yaml(tmp_path / "config.yaml", "workspace_dir: my-out/\n")
        settings = load_settings(config_path=cfg)
        assert Path(settings.state_dir) == Path("my-out/runs/legacy/state")
        assert Path(settings.reports_dir) == Path("my-out/runs/legacy/reports")
        assert Path(settings.memory_path) == Path("my-out/runs/legacy/memory/MEMORY.md")
        assert Path(settings.qa_dir) == Path("my-out/runs/legacy/qa")
        assert Path(settings.data.lake_dir) == Path("my-out/data_lake")

    def test_explicit_yaml_knobs_beat_derivation(self, tmp_path):
        cfg = _write_yaml(
            tmp_path / "config.yaml",
            """
            workspace_dir: ws/
            state_dir: elsewhere/state
            data:
              lake_dir: /abs/lake
            """,
        )
        settings = load_settings(config_path=cfg)
        assert Path(settings.state_dir) == Path("elsewhere/state")
        assert Path(settings.data.lake_dir) == Path("/abs/lake")
        # Knobs not explicitly set still derive — the run-scoped ones through the run dir.
        assert Path(settings.reports_dir) == Path("ws/runs/legacy/reports")
        assert Path(settings.memory_path) == Path("ws/runs/legacy/memory/MEMORY.md")

    def test_constructor_overrides_are_explicit_too(self, tmp_path):
        settings = load_settings(
            config_path=tmp_path / "missing.yaml",
            state_dir=str(tmp_path / "state"),
            reports_dir=str(tmp_path / "reports"),
        )
        assert Path(settings.state_dir) == tmp_path / "state"
        assert Path(settings.reports_dir) == tmp_path / "reports"
        # An untouched knob keeps deriving from the (default) run dir.
        assert Path(settings.memory_path) == Path("workspace/runs/legacy/memory/MEMORY.md")

    def test_nested_lake_dir_in_yaml_survives_derivation(self, tmp_path):
        """A ``data:`` block that sets OTHER keys must not lose the derived lake_dir."""
        cfg = _write_yaml(
            tmp_path / "config.yaml",
            """
            workspace_dir: ws/
            data:
              budget_usd: 9.0
            """,
        )
        settings = load_settings(config_path=cfg)
        assert settings.data.budget_usd == 9.0
        assert Path(settings.data.lake_dir) == Path("ws/data_lake")


class TestLegacyLayoutGuard:
    """Un-migrated legacy artifacts must stop a run, not silently empty the board."""

    def _project(self, tmp_path, monkeypatch, *legacy: str) -> Path:
        """A project root: a config file plus optional legacy artifact dirs."""
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "config.yaml"
        cfg.write_text("mode: paper\n")
        for name in legacy:
            (tmp_path / name).mkdir(parents=True)
        return cfg

    def test_flags_each_orphaned_legacy_artifact(self, tmp_path, monkeypatch):
        from noctis.bootstrap import detect_legacy_layout

        cfg = self._project(tmp_path, monkeypatch, "state", "data_lake", "reports")
        found = detect_legacy_layout(load_settings(config_path=cfg))
        assert {Path(a.legacy).name for a in found} == {"state", "data_lake", "reports"}

    def test_flags_an_orphaned_legacy_memory_file(self, tmp_path, monkeypatch):
        from noctis.bootstrap import detect_legacy_layout

        cfg = self._project(tmp_path, monkeypatch)
        (tmp_path / "MEMORY.md").write_text("# MEMORY\n")
        found = detect_legacy_layout(load_settings(config_path=cfg))
        assert [Path(a.legacy).name for a in found] == ["MEMORY.md"]

    def test_silent_when_the_configured_counterpart_exists(self, tmp_path, monkeypatch):
        """(Before run-scoped state the counterpart was ``workspace/state``; now the configured
        location is the run's own ``state/``, so that is what makes a root artifact a non-orphan.)
        """
        from noctis.bootstrap import detect_legacy_layout

        cfg = self._project(tmp_path, monkeypatch, "state")
        (tmp_path / "workspace" / "runs" / "legacy" / "state").mkdir(parents=True)
        assert detect_legacy_layout(load_settings(config_path=cfg)) == []

    def test_silent_when_a_knob_explicitly_points_at_the_legacy_path(self, tmp_path, monkeypatch):
        from noctis.bootstrap import detect_legacy_layout

        cfg = self._project(tmp_path, monkeypatch, "state")
        cfg.write_text("mode: paper\nstate_dir: state/\n")
        assert detect_legacy_layout(load_settings(config_path=cfg)) == []

    def test_silent_on_a_fresh_layout(self, tmp_path, monkeypatch):
        from noctis.bootstrap import detect_legacy_layout

        cfg = self._project(tmp_path, monkeypatch)
        assert detect_legacy_layout(load_settings(config_path=cfg)) == []

    def test_report_refuses_beside_legacy_state_naming_migrate(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from noctis.cli import app

        cfg = self._project(tmp_path, monkeypatch, "state")
        result = CliRunner().invoke(app, ["report", "--config", str(cfg)])
        assert result.exit_code == 2
        assert "noctis migrate" in result.output

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["data", "status"], id="status"),
            pytest.param(["data", "sync"], id="sync"),
            pytest.param(
                ["data", "ingest", "AAPL", "--start", "2026-01-05", "--end", "2026-01-09"],
                id="ingest",
            ),
        ],
    )
    def test_the_lake_verbs_refuse_beside_legacy_state_naming_migrate(
        self, argv, tmp_path, monkeypatch
    ):
        """The three lake verbs read the workspace like every other reader, so they refuse the
        same way beside un-migrated data — including ``ingest`` and ``sync``, which would
        otherwise spend vendor money filling a lake the engine no longer looks at. The guard is
        the wrapper's now (story #298); this is the refusal an operator still gets."""
        from typer.testing import CliRunner

        from noctis.cli import app

        cfg = self._project(tmp_path, monkeypatch, "data_lake")
        result = CliRunner().invoke(app, [*argv, "--config", str(cfg)])
        assert result.exit_code == 2, result.output
        assert "noctis migrate" in result.output

    def test_status_warns_but_still_prints(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from noctis.cli import app

        cfg = self._project(tmp_path, monkeypatch, "state")
        result = CliRunner().invoke(app, ["status", "--config", str(cfg)])
        assert result.exit_code == 0
        assert "legacy" in result.output.lower()
        assert "mode (resolved):" in result.output


class TestReportsUnderWorkspace:
    def test_generated_report_lands_under_the_workspace(self, tmp_path, monkeypatch):
        """`noctis report` writes to the run's reports dir — no hardcoded location."""
        from typer.testing import CliRunner

        from noctis.cli import app

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "config.yaml"
        cfg.write_text("mode: paper\n")
        result = CliRunner().invoke(app, ["report", "--as-of", "2026-01-02", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        run = tmp_path / "workspace" / "runs" / "legacy"
        assert (run / "reports" / "2026-01-02.md").is_file()
        # The command built the agent memory too — under the run, not the project root.
        assert (run / "memory" / "MEMORY.md").is_file()
        assert not (tmp_path / "MEMORY.md").exists()


class TestMemorySeed:
    """First run seeds the workspace memory from the committed MEMORY.seed.md."""

    def _project(self, tmp_path, monkeypatch, *, seed: str | None) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("mode: paper\n")
        if seed is not None:
            (tmp_path / "MEMORY.seed.md").write_text(seed)

    def _build(self, tmp_path):
        from noctis.bootstrap import build_memory

        return build_memory(load_settings(config_path=tmp_path / "config.yaml"))

    def test_first_run_copies_the_seed_before_the_store_constructs(self, tmp_path, monkeypatch):
        self._project(tmp_path, monkeypatch, seed="# MEMORY\n\n## Learnings\n\n- seeded lesson\n")
        memory = self._build(tmp_path)
        live = tmp_path / "workspace" / "runs" / "legacy" / "memory" / "MEMORY.md"
        assert live.is_file()
        # The seed's lesson survived — the store did NOT win the race with its blank template.
        assert "seeded lesson" in live.read_text()
        assert "seeded lesson" in memory.read()

    def test_no_seed_still_boots_with_the_blank_template(self, tmp_path, monkeypatch):
        self._project(tmp_path, monkeypatch, seed=None)
        self._build(tmp_path)
        live = tmp_path / "workspace" / "runs" / "legacy" / "memory" / "MEMORY.md"
        assert live.is_file()
        assert "## Learnings" in live.read_text()

    def test_an_existing_memory_is_never_overwritten_by_the_seed(self, tmp_path, monkeypatch):
        self._project(tmp_path, monkeypatch, seed="# MEMORY\n\n## Learnings\n\n- seeded lesson\n")
        live = tmp_path / "workspace" / "runs" / "legacy" / "memory" / "MEMORY.md"
        live.parent.mkdir(parents=True)
        live.write_text("# MEMORY\n\n## Learnings\n\n- my accumulated lesson\n")
        self._build(tmp_path)
        content = live.read_text()
        assert "my accumulated lesson" in content
        assert "seeded lesson" not in content
