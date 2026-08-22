"""Research-only segments in one record — ``noctis research --resume <id>`` (story #137).

A research-only night belongs to the same run. ``noctis research --resume <address>`` appends a
segment to an existing run: the same lock, the same frozen config, the same run-scoped state and
memory, the same record — so a run's research hours and trials accumulate whether they came from
``noctis run`` or from a standalone observable session.

Everything asserted here is external: what the record on disk says, which paths the session was
handed, what the command printed, and what it exited with. The research *behaviour* is untouched
by this story — the agent session is faked at the composition root's own seam
(``bootstrap.build_research_session``), exactly as ``tests/test_cli.py`` already fakes it, so no
test here needs an API key or contacts any external service.

The headline is :func:`test_a_run_that_only_ever_researched_reports_traded_false_and_null_perf`:
a run that never traded is a **first-class** shape, not a degenerate one — it reports
``traded: false`` and a ``null`` performance block rather than zeros, so a website renders
"researching" instead of a fake flat 0% equity curve (epic D10, §5.6).
"""

from __future__ import annotations

import json
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.reporting.run_store import RUN_LOCK_NAME, RUN_RECORD_NAME

runner = CliRunner()


# ── the faked agent session (no key, no network — the same seam test_cli.py uses) ───────────


class _Budgets:
    name = "test-profile"
    max_iterations = 20


class _Toolbox:
    author_calls = 0
    backtests_run = 3


class _FakeSession:
    """One research session that does nothing but take measurable time and journal what it is
    told to. The seam is real (``build_research_session``); only the model behind it is not."""

    model = "anthropic/claude-fake"
    budgets = _Budgets()
    toolbox = _Toolbox()

    def __init__(self, settings, *, trials: int = 0, work_s: float = 0.05) -> None:
        self.settings = settings
        self._trials = trials
        self._work_s = work_s

    def run(self, *, max_iterations=None, stop_event=None):
        from noctis.engine.research import ResearchSummary

        time.sleep(self._work_s)  # so the segment's measured RESEARCH seconds are real, not 0
        for trial in range(self._trials):
            _journal_trial(Path(self.settings.state_dir), "alpha", lookback=10 + trial)
        return ResearchSummary(
            iterations=4,
            promotions=1,
            rejections=2,
            stopped_reason="agent_done",
            candidates=["alpha"],
        )


def _journal_trial(state_dir: Path, strategy: str, **params) -> None:
    """One trial in the run's own experiment journal — the exhaustion gate's ground truth, and
    the only place the record's cumulative trial count is ever read from."""
    from noctis.research.journal import ExperimentJournal
    from tests.test_champions import make_scorecard

    ExperimentJournal(state_dir).record_trial(
        strategy,
        source="sweep",
        symbols=["AAPL"],
        params=params,
        window={},
        card=make_scorecard(strategy, test_metric=1.2, train_metric=1.4),
    )


@pytest.fixture
def sessions(monkeypatch):
    """Make ``noctis research`` believe an agent session is available, and capture what it was
    built with — the settings a session is handed are the contract this story is about."""
    from noctis.research.llm import ClientStatus

    monkeypatch.setattr(
        "noctis.research.client_status",
        lambda settings: ClientStatus(
            ok=True, model="anthropic/claude-fake", provider="anthropic", reason=None
        ),
    )
    built: list[_FakeSession] = []
    knobs: dict[str, object] = {"trials": 0}

    def fake_build(**kwargs):
        session = _FakeSession(kwargs["settings"], trials=int(knobs["trials"]))
        built.append(session)
        return session

    monkeypatch.setattr("noctis.bootstrap.build_research_session", fake_build)
    return _Sessions(built, knobs)


class _Sessions:
    def __init__(self, built: list[_FakeSession], knobs: dict[str, object]) -> None:
        self.built = built
        self._knobs = knobs

    def journals(self, trials: int) -> None:
        """Every session built from here on journals ``trials`` trials into the run's state."""
        self._knobs["trials"] = trials

    @property
    def last(self) -> _FakeSession:
        return self.built[-1]


# ── the workspace under test ───────────────────────────────────────────────────────────────


def _config(tmp_path: Path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}")
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted(p for p in _runs_dir(tmp_path).iterdir() if p.is_dir())


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _echoed_run_id(result) -> str:
    """The id the kickoff banner names — ``Run:`` on a fresh run, ``Resumed run:`` on a resume."""
    (line,) = [
        text
        for text in result.output.splitlines()
        if text.startswith("Run: ") or text.startswith("Resumed run: ")
    ]
    return line.split(": ", 1)[1].strip()


def _seal(run_dir: Path) -> None:
    record = _record(run_dir)
    record["run"]["status"] = "completed"
    record["run"]["completed_utc"] = record["run"]["last_active_utc"]
    (run_dir / RUN_RECORD_NAME).write_text(json.dumps(record))


def _research(cfg: str, *args: str):
    return runner.invoke(app, ["research", "--config", cfg, *args])


# ── a research session is a run, and a research segment says so ────────────────────────────


def test_noctis_research_mints_its_own_run_and_records_a_research_segment(tmp_path, sessions):
    """A standalone session is a run like any other: its own id, its own tree, its own record —
    not an unrecorded write into the reserved default run (which is what it used to be: state
    landed under `legacy` while `noctis runs` listed that run as record-less)."""
    from noctis.config.settings import DEFAULT_RUN_ID

    result = _research(_config(tmp_path))

    assert result.exit_code == 0, result.output
    (run_dir,) = _run_dirs(tmp_path)
    assert run_dir.name != DEFAULT_RUN_ID
    record = _record(run_dir)
    assert record["run"]["run_id"] == run_dir.name == _echoed_run_id(result)
    assert len(record["segments"]) == 1
    assert record["segments"][0]["command"] == "research"
    assert record["segments"][0]["resumed"] is False
    # …and the session's own state went into that run, not into the reserved default.
    assert Path(sessions.last.settings.state_dir) == run_dir / "state"


def test_a_research_segment_carries_its_own_stamps_reason_argv_and_counters(tmp_path, sessions):
    result = _research(_config(tmp_path), "-v")

    assert result.exit_code == 0, result.output
    (segment,) = _record(_run_dirs(tmp_path)[0])["segments"]
    assert segment["started_utc"].endswith("Z") and segment["stopped_utc"].endswith("Z")
    assert segment["duration_s"] >= 0.0
    assert segment["stopped_reason"] == "agent_done"  # the session's own reason, not the loop's
    assert segment["status"] == "stopped"
    # argv is this process's own (the runner's, under test) — that it is recorded at all is the
    # contract; what a `research` invocation records is pinned in tests/test_run_store.py.
    assert isinstance(segment["argv"], list)
    assert segment["counters"] == {
        "sessions": 1,
        "research_iterations": 4,
        "research_promotions": 1,
    }
    assert segment["phase_seconds"]["RESEARCH"] > 0.0


def test_a_research_minted_run_freezes_the_safety_gates_verdict(tmp_path, sessions):
    """A research session places no orders, but the run it mints may trade on a later segment —
    so the gate is resolved and its verdict frozen at creation like any other run's (#247).
    ``null`` is left to mean the one thing it should: an adopted history that froze no verdict."""
    result = _research(_config(tmp_path))

    assert result.exit_code == 0, result.output
    record = _record(_run_dirs(tmp_path)[0])
    assert record["inputs"]["execution_mode"] == "paper"
    assert record["assumptions"]["paper_only"] is True


def test_a_research_minted_runs_results_cannot_acquire_live_segments(
    tmp_path, sessions, monkeypatch
):
    """The frozen verdict is a real one, so it binds every later segment: a run minted by
    ``research`` refuses a live ``run --resume`` exactly as a run-minted one does (#247)."""
    cfg = _config(tmp_path)
    run_id = _echoed_run_id(_research(cfg))
    Path(cfg).write_text(f"mode: live\ndata:\n  lake_dir: {tmp_path}/lake\n")
    monkeypatch.setenv("ALLOW_LIVE", "true")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert result.exit_code == 1
    assert "paper" in result.output and "live" in result.output
    assert len(_record(_runs_dir(tmp_path) / run_id)["segments"]) == 1  # refused before opening


def test_noctis_research_without_resume_mints_a_fresh_run_every_time(tmp_path, sessions):
    """Identity is minted, never derived: two sessions under one config are two runs."""
    cfg = _config(tmp_path)

    first = _research(cfg)
    second = _research(cfg)

    assert first.exit_code == 0 and second.exit_code == 0, second.output
    assert len(_run_dirs(tmp_path)) == 2
    assert _echoed_run_id(first) != _echoed_run_id(second)


def test_a_session_that_cannot_start_still_closes_its_segment_and_releases_the_lock(tmp_path):
    """No LLM, no session — but the invocation still happened, so the record says so and the
    next one is not blocked by a dangling lock.

    The early exit leaves through the band (#251): the ``typer.Exit`` unwinds the ``with``, which
    closes the segment with the reason the command reported and releases the lock. Nothing ran,
    so nothing is counted — "measured nothing", never a zeroed session.
    """
    result = _research(_config(tmp_path))

    assert result.exit_code == 1
    assert "[llm] extra" in result.output
    (run_dir,) = _run_dirs(tmp_path)
    assert not (run_dir / RUN_LOCK_NAME).exists()
    (segment,) = _record(run_dir)["segments"]
    assert segment["status"] == "stopped"
    assert segment["stopped_reason"] == "no_session"
    assert segment["counters"] == {}
    assert segment["phase_seconds"] is None


# ── resuming: the same run, one more segment ───────────────────────────────────────────────


def test_noctis_research_resume_appends_a_research_only_segment_to_an_existing_run(
    tmp_path, sessions
):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)

    result = _research(cfg, "--resume", run_dir.name)

    assert result.exit_code == 0, result.output
    assert [p.name for p in _run_dirs(tmp_path)] == [run_dir.name]  # nothing new was minted
    record = _record(run_dir)
    assert [s["index"] for s in record["segments"]] == [0, 1]
    assert [s["command"] for s in record["segments"]] == ["run", "research"]
    assert [s["resumed"] for s in record["segments"]] == [False, True]
    assert _echoed_run_id(result) == run_dir.name


def test_a_resumed_research_session_runs_under_the_runs_frozen_config(tmp_path, sessions):
    """The mandate and the metric a run was created under decide what its accumulated evidence
    means, so an edit between segments cannot re-steer it."""
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nresearch:\n  min_trials: 20\n")
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)
    Path(cfg).write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n"
        "promotion:\n  metric: total_return\nresearch:\n  min_trials: 2\n"
    )

    result = _research(cfg, "--resume", run_dir.name)

    assert result.exit_code == 0, result.output
    settings = sessions.last.settings
    assert settings.promotion.metric == "sortino"
    assert settings.research.min_trials == 20


def test_a_resumed_research_session_reads_the_runs_state_memory_and_strategy_tiers(
    tmp_path, sessions
):
    from noctis.strategies.library import LibraryPaths

    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)

    result = _research(cfg, "--resume", run_dir.name)

    assert result.exit_code == 0, result.output
    settings = sessions.last.settings
    assert Path(settings.run_dir) == run_dir
    assert Path(settings.state_dir) == run_dir / "state"
    assert Path(settings.memory_path) == run_dir / "memory" / "MEMORY.md"
    assert LibraryPaths.from_settings(settings).tmp == run_dir / "strategies" / "__tmp"
    # …while the shared lake stays workspace-level, never following a run.
    assert Path(settings.data.lake_dir) == tmp_path / "lake"


def test_a_resumed_research_session_refuses_to_be_re_steered(tmp_path, sessions):
    """Same refusals as ``run --resume``: the mandate and the metric are frozen at creation."""
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)

    directed = _research(cfg, "--resume", run_dir.name, "--directive", "do something else")
    scored = _research(cfg, "--resume", run_dir.name, "--metric", "total_return")

    assert directed.exit_code == 1 and "frozen" in directed.output
    assert scored.exit_code == 1 and "frozen" in scored.output
    assert len(_record(run_dir)["segments"]) == 1  # neither refusal opened a segment


# ── the lock, and the terminal state: identical to `run --resume` ───────────────────────────


def test_a_live_locked_run_refuses_a_research_resume(tmp_path, sessions):
    """Two engines writing one run would corrupt it, whichever verb they were started by."""
    from noctis.reporting.run_record import utc_iso

    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)
    (run_dir / RUN_LOCK_NAME).write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "pid": __import__("os").getpid(),  # alive on this host: never a stale lock
                "hostname_hash": _host_hash(),
                "started_utc": utc_iso(datetime.now(UTC)),
                "heartbeat_utc": utc_iso(datetime.now(UTC)),
            }
        )
    )

    result = _research(cfg, "--resume", run_dir.name)

    assert result.exit_code == 1
    assert "RUN LOCKED" in result.output
    assert len(_record(run_dir)["segments"]) == 1


def _host_hash() -> str:
    import hashlib
    import socket

    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def test_a_completed_run_refuses_a_research_resume(tmp_path, sessions):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    (run_dir,) = _run_dirs(tmp_path)
    _seal(run_dir)

    result = _research(cfg, "--resume", run_dir.name)

    assert result.exit_code == 1
    assert "completed" in result.output
    assert len(_record(run_dir)["segments"]) == 1
    assert len(_run_dirs(tmp_path)) == 1  # a refused resume never mints a run instead


# ── addressing: the same four forms, resolved by the same resolver ─────────────────────────


def test_research_resume_addresses_a_run_by_id_latest_a_record_path_and_a_label(tmp_path, sessions):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"])
    (run_dir,) = _run_dirs(tmp_path)

    addresses = [
        run_dir.name,
        "latest",
        str(run_dir / RUN_RECORD_NAME),
        "@nightly-momo",
    ]
    for address in addresses:
        result = _research(cfg, "--resume", address)
        assert result.exit_code == 0, f"{address}: {result.output}"
        assert _echoed_run_id(result) == run_dir.name

    record = _record(run_dir)
    assert len(record["segments"]) == 1 + len(addresses)
    assert {s["command"] for s in record["segments"][1:]} == {"research"}


def test_an_unknown_research_resume_address_exits_nonzero_saying_how_to_find_runs(
    tmp_path, sessions
):
    result = _research(_config(tmp_path), "--resume", "@no-such-label")

    assert result.exit_code == 1
    assert "RESUME" in result.output
    assert "no-such-label" in result.output and "noctis runs" in result.output


# ── the totals: research hours and trials accumulate across research-only segments ─────────


def test_cumulative_research_seconds_and_trials_include_research_only_segments(tmp_path, sessions):
    cfg = _config(tmp_path)
    sessions.journals(2)
    first = _research(cfg)
    run_id = _echoed_run_id(first)

    sessions.journals(3)
    second = _research(cfg, "--resume", run_id)

    assert second.exit_code == 0, second.output
    record = _record(_runs_dir(tmp_path) / run_id)
    research_s = [s["phase_seconds"]["RESEARCH"] for s in record["segments"]]
    assert len(research_s) == 2 and all(seconds > 0.0 for seconds in research_s)
    assert record["run"]["cumulative_research_s"] == pytest.approx(sum(research_s))
    # Read off the run's own journals at write time — never a counter handed across processes.
    assert record["run"]["cumulative_trials"] == 5


# ── the headline: a research-only run is first-class ────────────────────────────────────────


def test_a_run_that_only_ever_researched_reports_traded_false_and_null_perf(tmp_path, sessions):
    """Many research segments, zero trades: ``traded: false`` and ``performance: null`` — not
    zeros — while the research evidence fills normally."""
    cfg = _config(tmp_path)
    sessions.journals(2)
    run_id = _echoed_run_id(_research(cfg))
    for _ in range(3):
        assert _research(cfg, "--resume", run_id).exit_code == 0

    record = _record(_runs_dir(tmp_path) / run_id)

    assert record["run"]["traded"] is False
    assert record["performance"] is None
    assert "performance" in record  # an explicit null, never an omitted key
    assert len(record["segments"]) == 4
    assert {s["command"] for s in record["segments"]} == {"research"}
    assert record["run"]["cumulative_research_s"] > 0.0
    assert record["run"]["cumulative_trials"] == 8
    assert sum(s["counters"]["research_iterations"] for s in record["segments"]) == 16
