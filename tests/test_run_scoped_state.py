"""Run-scoped state with a shared lake (story #131, epic #126).

Everything asserted here is external: where the engine is told to write, what one run can see of
another, and what a fresh run's memory starts from. The mechanism is the one that already existed
— per-artifact paths derived from a root through one subpath helper — re-pointed at a **run dir**,
so the tests drive it exactly as ``tests/test_workspace.py`` drives the workspace derivation.

The headline is :class:`TestRunIsolation`: two runs in one workspace, different mandates, and
neither sees the other's champions, paper account or working-tier drafts — while both read and
write the same data lake, because vendor data is expensive and run-neutral.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from noctis.config import load_settings
from noctis.config.settings import DEFAULT_RUN_ID


def _config(tmp_path: Path, body: str = "mode: paper\n") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def _settings(tmp_path: Path, body: str = "mode: paper\n"):
    return load_settings(config_path=_config(tmp_path, body))


class TestRunScopedDerivation:
    """``state_dir`` / ``reports_dir`` / ``qa_dir`` / ``memory_path`` come from the run dir."""

    def test_the_run_scoped_paths_derive_from_the_default_run_dir(self, tmp_path):
        settings = _settings(tmp_path)
        run_dir = tmp_path / "workspace" / "runs" / DEFAULT_RUN_ID
        assert Path(settings.run_dir) == run_dir
        assert Path(settings.state_dir) == run_dir / "state"
        assert Path(settings.reports_dir) == run_dir / "reports"
        assert Path(settings.qa_dir) == run_dir / "qa"
        assert Path(settings.memory_path) == run_dir / "memory" / "MEMORY.md"

    def test_the_lake_stays_workspace_level(self, tmp_path):
        settings = _settings(tmp_path)
        assert Path(settings.data.lake_dir) == tmp_path / "workspace" / "data_lake"

    def test_an_explicit_run_dir_moves_every_run_scoped_path_but_not_the_lake(self, tmp_path):
        settings = _settings(tmp_path, f"mode: paper\nrun_dir: {tmp_path}/elsewhere\n")
        assert Path(settings.state_dir) == tmp_path / "elsewhere" / "state"
        assert Path(settings.memory_path) == tmp_path / "elsewhere" / "memory" / "MEMORY.md"
        assert Path(settings.data.lake_dir) == tmp_path / "workspace" / "data_lake"

    def test_an_explicit_per_artifact_knob_still_beats_the_derivation(self, tmp_path):
        settings = _settings(tmp_path, "mode: paper\nstate_dir: elsewhere/state\n")
        assert Path(settings.state_dir) == Path("elsewhere/state")
        assert Path(settings.reports_dir) == Path(settings.run_dir) / "reports"

    def test_the_strategy_working_tiers_live_under_the_run_and_the_seeds_do_not(self, tmp_path):
        from noctis.strategies.library import LibraryPaths

        settings = _settings(tmp_path)
        tiers = LibraryPaths.from_settings(settings)
        run_dir = Path(settings.run_dir)
        assert tiers.tmp == run_dir / "strategies" / "__tmp"
        assert tiers.champions == run_dir / "strategies" / "champions"
        # The committed seeds are read-only input for every run — unchanged, workspace-independent.
        assert tiers.seeds == Path(settings.strategies_dir)


class TestBindingARun:
    """The composition root re-points the run-scoped paths once a run's id is minted."""

    def test_binding_repoints_every_derived_run_scoped_path(self, tmp_path):
        from noctis.config.settings import bind_run_dir

        settings = _settings(tmp_path)
        bind_run_dir(settings, tmp_path / "workspace" / "runs" / "20260727T000000Z-abc123")

        run_dir = tmp_path / "workspace" / "runs" / "20260727T000000Z-abc123"
        assert Path(settings.run_dir) == run_dir
        assert Path(settings.state_dir) == run_dir / "state"
        assert Path(settings.reports_dir) == run_dir / "reports"
        assert Path(settings.qa_dir) == run_dir / "qa"
        assert Path(settings.memory_path) == run_dir / "memory" / "MEMORY.md"
        # The lake never follows a run.
        assert Path(settings.data.lake_dir) == tmp_path / "workspace" / "data_lake"

    def test_binding_leaves_an_explicitly_configured_path_alone(self, tmp_path):
        from noctis.config.settings import bind_run_dir

        settings = _settings(tmp_path, "mode: paper\nstate_dir: elsewhere/state\n")
        bind_run_dir(settings, tmp_path / "workspace" / "runs" / "r1")
        assert Path(settings.state_dir) == Path("elsewhere/state")  # the operator's pin holds
        assert Path(settings.reports_dir) == tmp_path / "workspace" / "runs" / "r1" / "reports"

    def test_opening_a_run_binds_that_runs_own_paths(self, tmp_path):
        """The one composition-root change: every collaborator built after the run opens reads
        the run's own state, with no path arithmetic in any command body."""
        from noctis.bootstrap import open_run_store

        settings = _settings(tmp_path)
        store = open_run_store(settings, argv=["run"])
        try:
            assert Path(settings.run_dir) == store.run_dir
            assert Path(settings.state_dir) == store.run_dir / "state"
            assert Path(settings.memory_path) == store.run_dir / "memory" / "MEMORY.md"
        finally:
            store.close(reason="stopped")


def _crown(settings, family: str) -> None:
    """Put one champion on this run's board, through the same builder the engine uses."""
    from noctis.champions import PromotionRules, build_registry
    from tests.test_champions import make_scorecard

    build_registry(settings).consider(
        make_scorecard(family, test_metric=1.5, train_metric=1.8),
        PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0),
    )


def _open_account(settings, equity: float) -> None:
    from datetime import date

    from noctis.broker.paper import PaperBroker
    from noctis.broker.persistence import AccountStore

    store = AccountStore(Path(settings.state_dir) / "paper_account.json")
    store.save(PaperBroker(starting_cash=equity), date(2026, 7, 27))


class TestRunIsolation:
    """Two runs, one workspace, different mandates — one board each, one lake between them."""

    @pytest.fixture
    def two_runs(self, tmp_path):
        from noctis.bootstrap import open_run_store

        opened = []
        settings = []
        for label in ("momentum", "mean-reversion"):
            one = _settings(tmp_path)
            one.research.mandate = label  # the two runs are steered differently
            store = open_run_store(one, argv=["run"])
            store.close(reason="stopped")
            opened.append(store)
            settings.append(one)
        return settings, opened

    def test_neither_run_sees_the_others_champions(self, two_runs):
        from noctis.champions import build_registry

        (first, second), _ = two_runs
        _crown(first, "vol_breakout")
        _crown(second, "mean_revert")

        assert [e.family for e in build_registry(first).list()] == ["vol_breakout"]
        assert [e.family for e in build_registry(second).list()] == ["mean_revert"]

    def test_neither_run_sees_the_others_paper_account(self, two_runs):
        from noctis.broker.persistence import AccountStore

        (first, second), _ = two_runs
        _open_account(first, 100_000.0)
        _open_account(second, 250_000.0)

        def equity(settings):
            return AccountStore(Path(settings.state_dir) / "paper_account.json").summary().equity

        assert equity(first) == 100_000.0
        assert equity(second) == 250_000.0

    def test_neither_run_sees_the_others_working_tier_drafts(self, two_runs):
        from noctis.strategies.library import LibraryPaths, list_strategies

        (first, second), _ = two_runs
        _draft(first, "first_draft")
        _draft(second, "second_draft")

        def discovered(settings):
            listed = list_strategies(LibraryPaths.from_settings(settings))
            return {entry["name"] for entry in listed}

        assert "first_draft" in discovered(first) and "second_draft" not in discovered(first)
        assert "second_draft" in discovered(second) and "first_draft" not in discovered(second)

    def test_both_runs_share_the_one_data_lake(self, two_runs, tmp_path):
        (first, second), _ = two_runs
        shared = tmp_path / "workspace" / "data_lake"
        assert Path(first.data.lake_dir) == shared
        assert Path(second.data.lake_dir) == shared

    def test_each_run_owns_its_own_tree_under_the_workspace(self, two_runs, tmp_path):
        (first, second), stores = two_runs
        assert first.run_dir != second.run_dir
        for settings, store in zip((first, second), stores, strict=True):
            assert Path(settings.state_dir).parent == store.run_dir
            assert store.run_dir.parent == tmp_path / "workspace" / "runs"


_DRAFT_HEADER = (
    '"""A throwaway draft.\n\nstatus: draft\nstyle: test\nsymbols: AAPL\ntuned: no\n"""\n'
)


def _draft(settings, name: str) -> Path:
    """Park one working-tier draft in this run's ``__tmp/`` — the tier discovery reads."""
    from noctis.strategies.library import LibraryPaths

    tmp = LibraryPaths.from_settings(settings).tmp
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{name}.py"
    path.write_text(_DRAFT_HEADER)
    return path


class TestPerRunMemory:
    """Each run starts from the committed seed and accumulates on its own."""

    @pytest.fixture
    def seeded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "MEMORY.seed.md").write_text("# MEMORY\n\n## Learnings\n\n- seeded lesson\n")
        return tmp_path

    def _run_memory(self, tmp_path, run_id: str):
        from noctis.bootstrap import build_memory
        from noctis.config.settings import bind_run_dir

        settings = _settings(tmp_path)
        bind_run_dir(settings, tmp_path / "workspace" / "runs" / run_id)
        return settings, build_memory(settings)

    def test_two_runs_each_start_from_the_seed_and_accumulate_independently(self, seeded):
        first_settings, first = self._run_memory(seeded, "run-a")
        second_settings, second = self._run_memory(seeded, "run-b")

        assert "seeded lesson" in first.read()
        assert "seeded lesson" in second.read()

        first.append_finding("run A learned something")
        second.append_finding("run B learned something else")

        assert "run A learned something" in first.read()
        assert "run A learned something" not in second.read()
        assert "run B learned something else" in second.read()
        assert "run B learned something else" not in first.read()
        assert Path(first_settings.memory_path).parent.parent.name == "run-a"
        assert Path(second_settings.memory_path).parent.parent.name == "run-b"

    def test_the_committed_seed_is_never_mutated(self, seeded):
        before = (seeded / "MEMORY.seed.md").read_text()
        _, first = self._run_memory(seeded, "run-a")
        first.append_finding("a lesson that must not reach the seed")
        _, second = self._run_memory(seeded, "run-b")
        second.append_finding("nor this one")
        assert (seeded / "MEMORY.seed.md").read_text() == before


class TestRunScopedReports:
    """The per-day close report keeps its exact shape, under the run's own reports directory."""

    def test_the_close_report_lands_in_the_runs_reports_dir_with_its_usual_shape(self, tmp_path):
        from noctis.bootstrap import build_memory, open_run_store
        from noctis.champions import build_registry
        from noctis.engine.report_assembly import assemble_report
        from noctis.reporting import write_report, write_report_json

        settings = _settings(tmp_path)
        store = open_run_store(settings, argv=["run"])
        try:
            data = assemble_report(
                as_of="2026-07-27",
                mode="paper",
                registry=build_registry(settings),
                memory=build_memory(settings),
                state_dir=settings.state_dir,
            )
            markdown = write_report(data, settings.reports_dir)
            structured = write_report_json(data, settings.reports_dir)
        finally:
            store.close(reason="stopped")

        assert markdown == store.run_dir / "reports" / "2026-07-27.md"
        assert structured == store.run_dir / "reports" / "2026-07-27.json"
        assert markdown.read_text().startswith("# Close-of-day report — 2026-07-27")
        assert json.loads(structured.read_text())["as_of"] == "2026-07-27"


class TestNoctisRunEndToEnd:
    """`noctis run` writes into the run it minted — no command body does path arithmetic."""

    def _ready_lake_config(self, tmp_path) -> str:
        from noctis.data import MarketDataLake
        from noctis.data.types import to_ns
        from tests._data_helpers import MockVendor

        lake_dir = tmp_path / "lake"
        lake = MarketDataLake(lake_dir, MockVendor(), budget_usd=10_000.0, calendar="XNYS")
        lake.ensure_coverage(
            "EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns("2026-01-01"), to_ns("2026-12-31")
        )
        return str(
            _config(
                tmp_path,
                "mode: paper\n"
                "universe: [AAPL, MSFT]\n"
                "research_time_budget_minutes: 0\n"
                "data:\n"
                f"  lake_dir: {lake_dir}\n"
                "  dataset: EQUS.MINI\n",
            )
        )

    def test_a_run_puts_its_memory_and_state_under_its_own_tree(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from noctis.cli import app

        monkeypatch.chdir(tmp_path)
        cfg = self._ready_lake_config(tmp_path)

        result = CliRunner().invoke(app, ["run", "--config", cfg, "--time-limit-hours", "0"])

        assert result.exit_code == 0, result.output
        runs = tmp_path / "workspace" / "runs"
        minted = [p for p in runs.iterdir() if p.is_dir()]
        assert len(minted) == 1
        assert (minted[0] / "memory" / "MEMORY.md").is_file()
        assert (minted[0] / "run.json").is_file()
        # Nothing landed in the old shared locations, and the reserved run stayed untouched.
        assert not (tmp_path / "workspace" / "state").exists()
        assert not (tmp_path / "workspace" / "memory").exists()
        assert not (runs / DEFAULT_RUN_ID).exists()
