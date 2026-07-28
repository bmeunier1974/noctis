"""``noctis report <address>`` — one run's close-of-day report, by address (story #148).

A bare ``noctis report`` reads the reserved ``legacy`` run, which is what an invocation that never
opened a run *should* read — and is never the run ``noctis run`` just minted. So ``report`` learns
the same four address forms every other verb that names a run already takes, resolved by the same
resolver (``run_store.resolve_run_dir``), because an address form invented twice would eventually
resolve two different runs from one string.

Everything asserted here is external: what the command prints, what it exits with, and which run's
``reports/`` directory a generated report lands in. Runs are minted by the real CLI rather than
faked on disk, so the addresses under test are the ones an operator would actually type.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from noctis.cli import app

runner = CliRunner()

REPORT_DAY = "2026-07-27"


def _config(tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n")
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _mint_run(tmp_path: Path, cfg: str, *, label: str | None = None) -> Path:
    """One real ``noctis run`` — the run tree an operator would then address."""
    argv = ["run", "--config", cfg] + (["--label", label] if label else [])
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    minted = sorted(
        (p for p in _runs_dir(tmp_path).iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return minted[-1]


def _write_report(run_dir: Path, day: str, body: str) -> Path:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"{day}.md"
    path.write_text(body)
    return path


def _legacy_reports(tmp_path: Path) -> Path:
    return _runs_dir(tmp_path) / "legacy" / "reports"


def _seed_champion(run_dir: Path, family: str) -> None:
    """Crown one champion on this run's own board — the state a generated report reads."""
    from noctis.backtest.scorecard import Scorecard
    from noctis.champions.registry import ChampionEntry, ChampionRegistry

    registry = ChampionRegistry(run_dir / "state" / "champions.json", 3)
    params = {"fast": 3, "slow": 8}
    registry.champions.append(
        ChampionEntry(
            family=family,
            params=params,
            scorecard=Scorecard(family=family, params=params),
            crowned_at="2026-01-01",
            rationale="seed",
        )
    )
    registry.save()


# ── the four address forms ─────────────────────────────────────────────────────────────────


def test_report_at_a_run_id_prints_that_runs_close_of_day_report(tmp_path):
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _write_report(run_dir, REPORT_DAY, "# the addressed run's report\n")

    result = runner.invoke(app, ["report", run_dir.name, "--config", cfg])

    assert result.exit_code == 0, result.output
    assert "the addressed run's report" in result.output


def test_report_takes_the_same_address_forms_as_resume(tmp_path):
    """One resolver, one set of rules: a verb that addresses a run understands every form."""
    cfg = _config(tmp_path)
    other = _mint_run(tmp_path, cfg)
    _write_report(other, REPORT_DAY, "# the other run's report\n")
    momo = _mint_run(tmp_path, cfg, label="nightly-momo")
    _write_report(momo, REPORT_DAY, "# nightly-momo's report\n")

    for address in (momo.name, "@nightly-momo", "latest", str(momo / "run.json")):
        result = runner.invoke(app, ["report", address, "--config", cfg])
        assert result.exit_code == 0, f"{address}: {result.output}"
        assert "nightly-momo's report" in result.output, address
        assert "the other run's report" not in result.output, address


def test_report_without_an_address_still_reads_the_reserved_legacy_run(tmp_path):
    """No address keeps today's behaviour exactly: the run every unaddressed verb reads."""
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _write_report(run_dir, REPORT_DAY, "# the addressed run's report\n")
    legacy = _legacy_reports(tmp_path)
    legacy.mkdir(parents=True)
    (legacy / f"{REPORT_DAY}.md").write_text("# the legacy run's report\n")

    result = runner.invoke(app, ["report", "--config", cfg])

    assert result.exit_code == 0, result.output
    assert "the legacy run's report" in result.output
    assert "the addressed run's report" not in result.output


def test_report_and_as_of_compose_on_the_addressed_run(tmp_path):
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _write_report(run_dir, REPORT_DAY, "# the day asked for\n")
    _write_report(run_dir, "2026-07-28", "# a later day\n")

    result = runner.invoke(app, ["report", "latest", "--as-of", REPORT_DAY, "--config", cfg])

    assert result.exit_code == 0, result.output
    assert "the day asked for" in result.output
    assert "a later day" not in result.output


# ── refusals: the resolver's contract, unchanged ───────────────────────────────────────────


def test_report_at_an_unknown_address_exits_nonzero_naming_the_address(tmp_path):
    cfg = _config(tmp_path)
    _mint_run(tmp_path, cfg)

    result = runner.invoke(app, ["report", "20260101T000000Z-nope00", "--config", cfg])

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


def test_report_at_an_ambiguous_label_refuses_naming_both_runs(tmp_path):
    """A label may be reassigned; the id is the identity. Two answers is a refusal, not a pick."""
    cfg = _config(tmp_path)
    first = _mint_run(tmp_path, cfg, label="nightly-momo")
    second = _mint_run(tmp_path, cfg, label="nightly-momo")
    _write_report(first, REPORT_DAY, "# first\n")
    _write_report(second, REPORT_DAY, "# second\n")

    result = runner.invoke(app, ["report", "@nightly-momo", "--config", cfg])

    assert result.exit_code == 1
    assert first.name in result.output and second.name in result.output


# ── what an addressed run reads and writes ─────────────────────────────────────────────────


def test_a_run_with_no_report_yet_gets_one_assembled_from_that_runs_own_state(tmp_path):
    """Generation-on-miss follows the address: the run's own state in, the run's own reports out —
    nothing is assembled from, or written into, the reserved legacy tree."""
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _seed_champion(run_dir, "addressed_run_champion")
    _seed_champion(_runs_dir(tmp_path) / "legacy", "legacy_run_champion")

    result = runner.invoke(app, ["report", run_dir.name, "--as-of", REPORT_DAY, "--config", cfg])

    assert result.exit_code == 0, result.output
    assert f"# Close-of-day report — {REPORT_DAY}" in result.output
    assert "addressed_run_champion" in result.output
    assert "legacy_run_champion" not in result.output
    assert (run_dir / "reports" / f"{REPORT_DAY}.md").is_file()
    assert not _legacy_reports(tmp_path).exists()


def test_a_pruned_run_refuses_rather_than_inventing_a_report_from_deleted_state(tmp_path):
    """Retention deleted this run's state and reports on purpose, and the record says so. A report
    assembled from what is left would claim an empty champion board for a run that had one, so the
    honest answer is a refusal that points at the record retention kept."""
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _write_report(run_dir, REPORT_DAY, "# the report retention removed\n")
    sealed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--finish"])
    assert sealed.exit_code == 0, sealed.output
    pruned = runner.invoke(app, ["run-prune", run_dir.name, "--config", cfg])
    assert pruned.exit_code == 0, pruned.output

    result = runner.invoke(app, ["report", run_dir.name, "--config", cfg])

    assert result.exit_code == 1
    assert run_dir.name in result.output
    assert "prune" in result.output.lower()
    assert "run-record" in result.output
    assert not (run_dir / "reports").exists()  # nothing was resurrected
    assert json.loads((run_dir / "run.json").read_text())["run"]["state_pruned"] is True


def test_an_address_is_authoritative_beside_an_un_migrated_legacy_layout(tmp_path, monkeypatch):
    """The legacy guard answers "which tree does an *unaddressed* command read?" — and an address
    answers it instead, so a named run still prints beside a pre-workspace layout that refuses a
    bare ``report``."""
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    _write_report(run_dir, REPORT_DAY, "# the addressed run's report\n")
    (tmp_path / "reports").mkdir()  # pre-workspace artifacts beside config.yaml
    (tmp_path / "state").mkdir()

    bare = runner.invoke(app, ["report", "--config", cfg])
    addressed = runner.invoke(app, ["report", run_dir.name, "--config", cfg])

    assert bare.exit_code == 2  # unaddressed: still the refusal that names `noctis migrate`
    assert "noctis migrate" in bare.output
    assert addressed.exit_code == 0, addressed.output
    assert "the addressed run's report" in addressed.output


def test_the_address_argument_is_documented_in_the_house_wording(tmp_path):
    result = runner.invoke(app, ["report", "--help"])

    assert result.exit_code == 0, result.output
    assert "Run address" in result.output
    assert "four forms" in result.output
