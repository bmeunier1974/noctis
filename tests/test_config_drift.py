"""Config drift — ``--show-config-drift`` and ``--rebase-config`` (story #134, epic #126).

A run's configuration is frozen at creation (story #132) and drift is normal: edit ``config.yaml``
or a mandate profile between segments and the frozen values simply keep winning. This story is the
two things an operator needs on top of that — **see** what the current files would change, and
**adopt** them deliberately when they mean to.

The two halves have opposite postures, and both are asserted here as external behaviour:

* ``--show-config-drift`` is an *inspection*. It prints a diff and exits, and it must leave the run
  exactly as it found it — no segment, no lock, not one byte of the record rewritten.
* ``--rebase-config`` is a *decision*. It adopts the current files, bumps ``inputs.config_epoch``
  and appends a before/after entry to ``inputs.config_changes`` naming the segment it happened in,
  so a record whose config changed mid-run says so and says where. With nothing to adopt it is a
  documented no-op: the epoch never moves for nothing.

The refused tier is absolute in both: ``mode`` and ``allow_live`` are never recorded, never
restored and never rebasable, and asking for it is refused with a message rather than ignored.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from noctis.cli import app
from noctis.reporting.run_store import RUN_LOCK_NAME, RUN_RECORD_NAME
from noctis.reporting.schema import validate

runner = CliRunner()


def _config(tmp_path: Path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}")
    return str(path)


def _rewrite(cfg: str, tmp_path: Path, body: str) -> None:
    Path(cfg).write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}"
    )


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted(p for p in _runs_dir(tmp_path).iterdir() if p.is_dir())


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _start(tmp_path: Path, cfg: str) -> Path:
    """Night one: mint a run under ``cfg`` and return its directory."""
    result = runner.invoke(app, ["run", "--config", cfg])
    assert result.exit_code == 0, result.output
    return _run_dirs(tmp_path)[0]


def _mandate(tmp_path: Path, body: str) -> Path:
    profiles = tmp_path / "mandate" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    profile = profiles / "aggressive.md"
    profile.write_text(textwrap.dedent(body))
    return profile


# ── --show-config-drift: see it, change nothing ────────────────────────────────────────────


def test_show_config_drift_prints_a_readable_diff_of_the_frozen_keys(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nchampion_count: 5\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert result.exit_code == 0, result.output
    assert "promotion.metric" in result.output
    assert "sortino" in result.output and "total_return" in result.output
    assert "champion_count" in result.output
    assert run_dir.name in result.output


def test_show_config_drift_opens_no_segment_takes_no_lock_and_rewrites_nothing(tmp_path):
    """An inspection command that mutated the thing it inspects would be the worst of both: it
    would make "let me look first" itself a decision."""
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)
    before = (run_dir / RUN_RECORD_NAME).read_bytes()
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / RUN_RECORD_NAME).read_bytes() == before
    assert not (run_dir / RUN_LOCK_NAME).exists()
    assert len(_record(run_dir)["segments"]) == 1
    assert _run_dirs(tmp_path) == [run_dir]  # and it never mints a run of its own


def test_show_config_drift_reports_a_rewritten_mandate_profile_as_drift(tmp_path):
    """Mandate drift is drift in the resolved *text*, which is exactly why it is frozen as text:
    the selector never moved, and the bytes behind it did."""
    _mandate(tmp_path, "---\nsummary: hunt volatility\n---\n\nTrade the most volatile names.\n")
    cfg = _config(tmp_path, f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n")
    run_dir = _start(tmp_path, cfg)
    _mandate(tmp_path, "---\nsummary: something else\n---\n\nBuy and hold index funds.\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert result.exit_code == 0, result.output
    assert "mandate" in result.output.lower()
    assert "index funds" in result.output
    assert "most volatile" in result.output


def test_a_live_tier_edit_is_not_reported_as_drift(tmp_path):
    """The live tier is this process's by design — a per-process budget or a path that moved is
    not something to "adopt", so offering it would be noise that hides the real difference."""
    cfg = _config(tmp_path, "time_limit_hours: 1.0\nqa:\n  keep_last_runs: 3\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="time_limit_hours: 9.0\nqa:\n  keep_last_runs: 40\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert result.exit_code == 0, result.output
    assert "time_limit_hours" not in result.output
    assert "qa.keep_last_runs" not in result.output
    assert "No config drift" in result.output


def test_a_run_with_no_drift_reports_none(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert result.exit_code == 0, result.output
    assert "No config drift" in result.output


def test_show_config_drift_without_a_resume_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path), "--show-config-drift"])

    assert result.exit_code == 1
    assert "--resume" in result.output


def test_show_config_drift_and_rebase_config_together_are_refused(tmp_path):
    cfg = _config(tmp_path)
    run_dir = _start(tmp_path, cfg)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            cfg,
            "--resume",
            run_dir.name,
            "--show-config-drift",
            "--rebase-config",
        ],
    )

    assert result.exit_code == 1
    assert "--show-config-drift" in result.output and "--rebase-config" in result.output


def test_show_config_drift_on_an_unknown_run_exits_nonzero(tmp_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            _config(tmp_path),
            "--resume",
            "20260101T000000Z-nope00",
            "--show-config-drift",
        ],
    )

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


# ── the frozen config keeps winning while the drift is only *reported* ─────────────────────


def test_resuming_without_rebase_leaves_frozen_winning_while_the_flag_reports_the_difference(
    tmp_path,
):
    """The story's headline: edit both the config and the mandate profile between segments, resume
    normally, and nothing moves — same frozen values, same digest, same mandate text — while
    ``--show-config-drift`` still shows exactly what was not adopted."""
    _mandate(tmp_path, "---\nsummary: hunt volatility\n---\n\nTrade the most volatile names.\n")
    cfg = _config(
        tmp_path,
        f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n"
        "promotion:\n  metric: sortino\nchampion_count: 5\n",
    )
    run_dir = _start(tmp_path, cfg)
    digest_before = _record(run_dir)["inputs"]["settings"]["digest"]
    _mandate(tmp_path, "---\nsummary: something else\n---\n\nBuy and hold index funds.\n")
    _rewrite(
        tmp_path=tmp_path,
        cfg=cfg,
        body=f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n"
        "promotion:\n  metric: total_return\nchampion_count: 1\n",
    )

    resumed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])
    drift = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--show-config-drift"]
    )

    assert resumed.exit_code == 0, resumed.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["settings"]["resolved"]["promotion"]["metric"] == "sortino"
    assert frozen["settings"]["resolved"]["champion_count"] == 5
    assert frozen["settings"]["digest"] == digest_before
    assert frozen["mandate"]["text"] == "Trade the most volatile names."
    assert frozen["config_epoch"] == 1
    assert frozen["config_changes"] == []
    assert drift.exit_code == 0, drift.output
    assert "promotion.metric" in drift.output and "champion_count" in drift.output
    assert "index funds" in drift.output


# ── --rebase-config: adopt it deliberately ─────────────────────────────────────────────────


def test_rebase_config_adopts_the_current_config_bumps_the_epoch_and_records_the_change(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nchampion_count: 5\n")
    run_dir = _start(tmp_path, cfg)
    digest_before = _record(run_dir)["inputs"]["settings"]["digest"]
    _rewrite(
        tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\nchampion_count: 1\n"
    )

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["settings"]["resolved"]["promotion"]["metric"] == "total_return"
    assert frozen["settings"]["resolved"]["champion_count"] == 1
    assert frozen["config_epoch"] == 2
    (change,) = frozen["config_changes"]
    assert change["from_epoch"] == 1 and change["to_epoch"] == 2
    assert change["segment"] == 1  # the segment that adopted it, so the record says *where*
    assert change["at"].endswith("Z")
    assert change["digest_before"] == digest_before
    assert change["digest_after"] == frozen["settings"]["digest"] != digest_before
    assert {entry["path"] for entry in change["settings"]} == {
        "promotion.metric",
        "champion_count",
    }
    assert {"path": "champion_count", "from": 5, "to": 1} in change["settings"]


def test_a_rebased_record_stays_schema_valid(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    assert validate(_record(run_dir)) == []


def test_the_rebased_config_is_what_a_later_resume_restores(tmp_path):
    """Adoption is the whole point: after a rebase the *new* values are the run's own, so a third
    segment restores them even if the files move again."""
    from noctis.bootstrap import resolve_session

    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nchampion_count: 5\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(
        tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\nchampion_count: 1\n"
    )
    rebased = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )
    assert rebased.exit_code == 0, rebased.output
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: sharpe\nchampion_count: 9\n")

    resumed = resolve_session(cfg, require_gate=True, resume=run_dir.name)

    assert resumed.settings.promotion.metric == "total_return"
    assert resumed.settings.champion_count == 1
    assert _record(run_dir)["inputs"]["config_epoch"] == 2


def test_rebase_config_adopts_a_rewritten_mandate_profile_as_the_runs_own_mandate(tmp_path):
    _mandate(tmp_path, "---\nsummary: hunt volatility\n---\n\nTrade the most volatile names.\n")
    cfg = _config(tmp_path, f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n")
    run_dir = _start(tmp_path, cfg)
    _mandate(tmp_path, "---\nsummary: something else\n---\n\nBuy and hold index funds.\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["mandate"]["text"] == "Buy and hold index funds."
    assert frozen["config_epoch"] == 2
    change = frozen["config_changes"][0]["mandate"]
    assert change["from"]["text_sha256"] != change["to"]["text_sha256"]


def test_rebase_config_on_a_drift_free_run_is_a_no_op_that_never_bumps_the_epoch(tmp_path):
    """A cosmetic bump would mark a run as mixed-config for nothing, and every consumer that
    renders ``config_epoch > 1`` as "this run changed mid-flight" would then be lying."""
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["config_epoch"] == 1
    assert frozen["config_changes"] == []
    assert len(_record(run_dir)["segments"]) == 2  # …and the resume itself still happened


def test_a_no_op_rebase_leaves_the_session_exactly_as_an_ordinary_resume_would(tmp_path):
    """A no-op has to mean the *session* too, not just the record. A mandate may bind per-process
    budgets, which are live tier — so resolving the current mandate to look for drift must not
    leave its overlay applied when there was nothing to adopt, or the same command would run under
    two different budgets depending on whether some unrelated key had moved."""
    from noctis.bootstrap import resolve_session

    _mandate(
        tmp_path,
        "---\nconfig:\n  time_limit_hours: 8.0\n---\n\nTrade the most volatile names.\n",
    )
    cfg = _config(
        tmp_path,
        f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n"
        "time_limit_hours: 1.0\n",
    )
    run_dir = _start(tmp_path, cfg)

    plain = resolve_session(cfg, require_gate=True, resume=run_dir.name)
    rebased = resolve_session(cfg, require_gate=True, resume=run_dir.name, rebase_config=True)

    assert rebased.rebase is None
    assert rebased.settings.time_limit_hours == plain.settings.time_limit_hours == 1.0


def test_rebase_config_without_a_resume_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path), "--rebase-config"])

    assert result.exit_code == 1
    assert "--resume" in result.output


def test_two_rebases_leave_two_entries_and_epoch_three(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\n")
    runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"])
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: sharpe\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["config_epoch"] == 3
    assert [change["to_epoch"] for change in frozen["config_changes"]] == [2, 3]
    assert [change["segment"] for change in frozen["config_changes"]] == [1, 2]


# ── the refused tier: never rebasable, under any flag ──────────────────────────────────────


def test_rebasing_the_execution_mode_is_refused_not_silently_ignored(tmp_path, monkeypatch):
    """The one concrete way to *attempt* rebasing the mode: edit ``mode`` in ``config.yaml``, open
    ``ALLOW_LIVE``, and ask for the current files. ``mode`` is not in the frozen settings at all —
    the record keeps only the gate's verdict — so the refusal is the mode check itself, and it says
    out loud that no flag lifts it."""
    cfg = _config(tmp_path)
    run_dir = _start(tmp_path, cfg)
    Path(cfg).write_text(f"mode: live\ndata:\n  lake_dir: {tmp_path}/lake\n")
    monkeypatch.setenv("ALLOW_LIVE", "true")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 1
    assert "--rebase-config" in result.output
    assert "mode" in result.output and "allow_live" in result.output
    record = _record(run_dir)
    assert len(record["segments"]) == 1  # refused before a segment opened
    assert record["inputs"]["config_epoch"] == 1


def test_a_rebased_record_still_carries_neither_live_money_gate(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\n")
    run_dir = _start(tmp_path, cfg)
    _rewrite(tmp_path=tmp_path, cfg=cfg, body="promotion:\n  metric: total_return\n")

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--rebase-config"]
    )

    assert result.exit_code == 0, result.output
    resolved = _record(run_dir)["inputs"]["settings"]["resolved"]
    assert "mode" not in resolved and "allow_live" not in resolved
    assert validate(_record(run_dir)) == []
