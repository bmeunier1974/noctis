"""The run-segment band — one context manager owns open → work → close (story #249, epic #246).

Everything asserted here is external: what ``run.json`` says once the segment closed, whether
``run.lock`` is gone, whether the QA tree was stamped, what the yielded handle exposes. Never a
call order and never a builder spy — the band is driven directly against a real store on a
``tmp_path`` runs directory with a fake clock, so a Typer command is never invoked to watch a
segment close.

Two contracts get particular attention, because they are the ones that used to be written twice
by hand in two command bodies:

* **Every exit path closes the segment and releases the lock** — a normal return, a body that
  never reported a reason (``"startup"``), a ``typer.Exit``, any exception, a recorder that
  refuses to build. A crash before the work starts must never leave a run locked.
* **"Measured nothing" is not "measured zero"** — a segment finished without an outcome writes no
  counters and no phase seconds, exactly as an early exit does today.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from noctis.bootstrap import SessionInputs, open_segment
from noctis.config import load_settings
from noctis.engine.research import ResearchSummary
from noctis.engine.runtime import RuntimeResult
from noctis.observability import Console, EventTee
from noctis.reporting.run_record import RESEARCH_PHASE
from noctis.reporting.run_store import RUN_LOCK_NAME, RunLockedError

START = datetime(2026, 8, 22, 13, 30, 0, tzinfo=UTC)


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _settings(tmp_path: Path, lines: list[str] | None = None, name: str = "config.yaml"):
    """Settings whose workspace (and therefore ``runs_dir``/``qa_dir``) lives under ``tmp_path``."""
    cfg = tmp_path / name
    cfg.write_text("\n".join(["mode: paper", *(lines or [])]) + "\n", encoding="utf-8")
    return load_settings(config_path=str(cfg))


def _inputs(settings, **fields) -> SessionInputs:
    """One resolved session, as ``run`` and ``research`` hand it to the composition root."""
    return SessionInputs(
        settings=settings,
        mode=fields.pop("mode", "paper"),
        mandate=fields.pop("mandate", None),
        overrides=fields.pop("overrides", []),
        **fields,
    )


def _llm(monkeypatch, *, ok: bool) -> None:
    """Say whether the configured research client is buildable — without contacting a provider."""
    import noctis.research as research

    status = research.ClientStatus(
        ok=ok,
        model="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        reason=None if ok else "no ANTHROPIC_API_KEY in .env for provider 'anthropic'",
    )
    monkeypatch.setattr(research, "client_status", lambda _settings: status)


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _segment(run_dir: Path, index: int = -1) -> dict:
    return _record(run_dir)["segments"][index]


def _locked(run_dir: Path) -> bool:
    return (run_dir / RUN_LOCK_NAME).exists()


def _result(**fields) -> RuntimeResult:
    return RuntimeResult(
        cycles_completed=fields.pop("cycles_completed", 3),
        research_iterations=fields.pop("research_iterations", 7),
        research_promotions=fields.pop("research_promotions", 1),
        trades=fields.pop("trades", 12),
        stopped_reason=fields.pop("stopped_reason", "time_limit"),
        phase_seconds=fields.pop("phase_seconds", {"RESEARCH": 120.0, "TRADING": 60.0}),
        **fields,
    )


# ── every exit path closes the segment and releases the lock ──────────────────────────────


def test_a_body_that_raises_before_finish_closes_at_startup_and_releases_the_lock(tmp_path):
    """The sentinel: a process that crashed before it reported anything measured nothing, and the
    run it opened is a closed, resumable run rather than a dangling lock."""
    settings = _settings(tmp_path)
    seen: list = []

    with pytest.raises(RuntimeError, match="boom"):
        with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
            seen.append(seg)
            raise RuntimeError("boom")

    run_dir = seen[0].store.run_dir
    assert _segment(run_dir)["stopped_reason"] == "startup"
    assert _segment(run_dir)["status"] == "stopped"
    assert not _locked(run_dir)


def test_a_body_that_raises_after_finish_keeps_the_reason_it_reported(tmp_path):
    """A reported reason is the segment's reason whatever happens next — the crash is what the
    record's ``interrupted`` marking and its events are for, not a reason nobody chose."""
    settings = _settings(tmp_path)
    seen: list = []

    with pytest.raises(RuntimeError, match="late"):
        with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
            seen.append(seg)
            seg.finish("no_data")
            raise RuntimeError("late")

    run_dir = seen[0].store.run_dir
    assert _segment(run_dir)["stopped_reason"] == "no_data"
    assert not _locked(run_dir)


def test_a_typer_exit_inside_the_body_still_closes_the_segment(tmp_path):
    """``research``'s "no session" path raises ``typer.Exit`` through the band. The TEST may
    import Typer to raise it; the band must not — it only has to unwind cleanly."""
    import typer

    settings = _settings(tmp_path)
    seen: list = []

    with pytest.raises(typer.Exit):
        with open_segment(
            _inputs(settings), command="research", argv=["research"], clock=FakeClock()
        ) as seg:
            seen.append(seg)
            seg.finish("no_session")
            raise typer.Exit(code=1)

    run_dir = seen[0].store.run_dir
    assert _segment(run_dir)["stopped_reason"] == "no_session"
    assert not _locked(run_dir)


def test_a_body_that_returns_normally_closes_with_what_it_reported(tmp_path):
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        seg.finish("time_limit", outcome=_result())

    assert _segment(seg.store.run_dir)["stopped_reason"] == "time_limit"
    assert not _locked(seg.store.run_dir)


# ── what finish measures, and what it refuses to invent ───────────────────────────────────


def test_a_run_segment_finished_with_a_result_writes_four_counters_and_phase_seconds(tmp_path):
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        seg.finish("time_limit", outcome=_result())

    segment = _segment(seg.store.run_dir)
    assert segment["counters"] == {
        "cycles": 3,
        "research_iterations": 7,
        "research_promotions": 1,
        "trades": 12,
    }
    assert segment["phase_seconds"] == {"RESEARCH": 120.0, "TRADING": 60.0}


def test_a_segment_that_measured_nothing_writes_neither_counters_nor_phase_seconds(tmp_path):
    """``outcome=None`` means nobody measured — the keys are absent, never zero-filled. A zero
    would be a claim about work this process did, made by a process that never did any."""
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        seg.finish("no_data")

    segment = _segment(seg.store.run_dir)
    assert segment["counters"] == {}
    assert segment["phase_seconds"] is None


def test_a_research_segment_writes_one_session_its_counters_and_no_trades_key(tmp_path):
    """A research session cannot place an order and the record derives ``run.traded`` from that
    counter, so the key is absent rather than a confidently false zero."""
    settings = _settings(tmp_path)
    summary = ResearchSummary(iterations=9, promotions=2, stopped_reason="agent_done")

    with open_segment(
        _inputs(settings), command="research", argv=["research"], clock=FakeClock()
    ) as seg:
        seg.finish("agent_done", outcome=summary, phase_seconds={RESEARCH_PHASE: 42.5})

    segment = _segment(seg.store.run_dir)
    assert segment["counters"] == {"sessions": 1, "research_iterations": 9, "research_promotions": 2}
    assert "trades" not in segment["counters"]
    assert segment["phase_seconds"] == {RESEARCH_PHASE: 42.5}


def test_a_second_finish_does_not_overwrite_the_first_reason(tmp_path):
    """The store's close is idempotent; the segment is too — the first reason a body chose is the
    one the record keeps."""
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        seg.finish("run_limit")
        seg.finish("time_limit", outcome=_result())

    segment = _segment(seg.store.run_dir)
    assert segment["stopped_reason"] == "run_limit"
    assert segment["counters"] == {}


def test_checkpoint_rewrites_the_record_before_the_segment_closes(tmp_path):
    """The runtime's ``on_cycle_close`` seam: a multi-week run's evidence is current on disk long
    before the process stops, so the counters are readable from inside the body."""
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        seg.checkpoint(_result(cycles_completed=1, trades=4))
        mid = _segment(seg.store.run_dir)
        assert mid["status"] == "running"
        assert mid["counters"]["cycles"] == 1
        assert mid["counters"]["trades"] == 4
        assert mid["phase_seconds"] == {"RESEARCH": 120.0, "TRADING": 60.0}
        seg.finish("time_limit", outcome=_result(cycles_completed=2, trades=9))

    assert _segment(seg.store.run_dir)["counters"]["cycles"] == 2


# ── what the band records on its own, and what it hands back ──────────────────────────────


def test_engine_notes_land_as_record_events_with_no_echo_callback(tmp_path):
    """Recording is the band's half of the engine-change note (D4): the record is what an
    experiment is judged from months later, so a note that only ever reached a terminal would be
    invisible exactly when it mattered — printing stays with the command."""
    settings = _settings(tmp_path)
    notes = ["engine_epoch 1 → 2: promotion moved", "engine_epoch 2 → 3: fills moved"]

    with open_segment(
        _inputs(settings, engine_notes=list(notes)),
        command="run",
        argv=["run"],
        clock=FakeClock(),
    ) as seg:
        seg.finish("time_limit", outcome=_result())

    recorded = [event["text"] for event in _record(seg.store.run_dir)["events"]]
    assert recorded == notes
    assert {event["segment"] for event in _record(seg.store.run_dir)["events"]} == {0}


def test_the_handle_reads_through_to_the_run_it_opened(tmp_path):
    settings = _settings(tmp_path)
    inputs = _inputs(settings)
    with open_segment(inputs, command="run", argv=["run"], label="tonight", clock=FakeClock()) as s:
        assert s.inputs is inputs
        assert s.command == "run"
        assert s.run_id == s.store.run_id
        assert s.record_path == s.store.record_path == s.store.run_dir / "run.json"
        assert s.resumed is False
        assert s.prior_runtime_s == 0.0
        s.finish("time_limit", outcome=_result())

    assert _record(s.store.run_dir)["run"]["label"] == "tonight"


# ── the resume the session carries, and the one fatal failure ─────────────────────────────


def test_the_sessions_resume_address_appends_a_segment_to_the_same_run(tmp_path):
    """``inputs.resume`` is the whole resume decision (D2): the address the operator typed, and
    ``resume is not None`` is the flag — so no entrypoint can carry the two apart."""
    settings = _settings(tmp_path)
    clock = FakeClock()
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=clock) as first:
        first.finish("time_limit", outcome=_result())
    run_id = first.run_id

    clock.advance(3600)
    with open_segment(
        _inputs(settings, resume=run_id), command="run", argv=["run", "--resume", run_id], clock=clock
    ) as second:
        assert second.run_id == run_id
        assert second.resumed is True
        assert second.prior_runtime_s > 0
        second.finish("stopped", outcome=_result())

    record = _record(second.store.run_dir)
    assert len(record["segments"]) == 2
    assert [s["index"] for s in record["segments"]] == [0, 1]
    assert record["segments"][1]["resumed"] is True


def test_a_live_lock_raises_run_locked_out_of_the_band(tmp_path):
    """Two engines writing one run is corruption, not degradation — so this is the one failure the
    band lets out. It is the typed error, never a Typer exit: mapping it to red text and an exit
    code is the command's job."""
    settings = _settings(tmp_path)
    clock = FakeClock()
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=clock) as held:
        with pytest.raises(RunLockedError):
            with open_segment(
                _inputs(settings, resume=held.run_id), command="run", argv=["run"], clock=clock
            ):
                pass
        held.finish("time_limit", outcome=_result())

    assert not _locked(held.store.run_dir)


# ── the --debug recorder, under each command's own rule (D5) ──────────────────────────────


def test_a_debug_run_segment_builds_a_recorder_on_the_runs_own_id(tmp_path, monkeypatch):
    _llm(monkeypatch, ok=True)
    settings = _settings(tmp_path)

    with open_segment(
        _inputs(settings), command="run", argv=["run", "--debug"], debug=True, clock=FakeClock()
    ) as seg:
        assert seg.recorder is not None
        assert seg.recorder.run_id == seg.run_id
        assert "legacy research loop" not in seg.recorder.funnel_line()
        seg.finish("time_limit", outcome=_result())


def test_a_run_segment_without_a_buildable_llm_marks_the_recorder_legacy(tmp_path):
    """The legacy proposer/Optuna loop instruments no funnel, so the recorder says so instead of
    rendering a comforting all-zeros funnel (AGENTS.md rule 2)."""
    settings = _settings(tmp_path)

    with open_segment(
        _inputs(settings), command="run", argv=["run", "--debug"], debug=True, clock=FakeClock()
    ) as seg:
        assert seg.recorder is not None
        assert seg.recorder.funnel_line() == "legacy research loop — funnel not instrumented"
        seg.finish("time_limit", outcome=_result())


def test_a_quiet_run_segment_builds_no_recorder(tmp_path):
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        assert seg.recorder is None
        seg.finish("time_limit", outcome=_result())


def test_a_debug_research_segment_without_a_buildable_llm_builds_no_recorder(tmp_path):
    """Research never records a legacy session, so an early exit leaves no orphaned half-written
    QA tree behind."""
    settings = _settings(tmp_path)

    with open_segment(
        _inputs(settings),
        command="research",
        argv=["research", "--debug"],
        debug=True,
        clock=FakeClock(),
    ) as seg:
        assert seg.recorder is None
        seg.finish("no_session")


def test_a_debug_research_segment_with_a_buildable_llm_builds_a_recorder(tmp_path, monkeypatch):
    _llm(monkeypatch, ok=True)
    settings = _settings(tmp_path)

    with open_segment(
        _inputs(settings),
        command="research",
        argv=["research", "--debug"],
        debug=True,
        clock=FakeClock(),
    ) as seg:
        assert seg.recorder is not None
        assert seg.recorder.run_id == seg.run_id
        seg.finish("agent_done")


def test_the_recorder_is_closed_when_the_body_raises(tmp_path, monkeypatch):
    _llm(monkeypatch, ok=True)
    settings = _settings(tmp_path)
    seen: list = []

    with pytest.raises(RuntimeError, match="boom"):
        with open_segment(
            _inputs(settings), command="run", argv=["run", "--debug"], debug=True, clock=FakeClock()
        ) as seg:
            seen.append(seg)
            raise RuntimeError("boom")

    manifest = json.loads((seen[0].recorder.run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["stopped"] is not None
    assert manifest["mode"] == "paper"


def test_a_recorder_that_refuses_to_build_still_closes_the_segment(tmp_path, monkeypatch):
    """The recorder is assembled *inside* the guarded region, so a failure there (an unwritable
    qa_dir, a prune that raised) can no longer leave the run locked until the stale-lock timeout."""
    import noctis.bootstrap as bootstrap

    def _refuse(*_args, **_kwargs):
        raise OSError("qa_dir is not writable")

    monkeypatch.setattr(bootstrap, "build_recorder", _refuse)
    settings = _settings(tmp_path)

    with pytest.raises(OSError, match="qa_dir"):
        with open_segment(
            _inputs(settings), command="run", argv=["run", "--debug"], debug=True, clock=FakeClock()
        ):
            pass

    run_dir = next((Path(settings.runs_dir)).iterdir())
    assert _segment(run_dir)["stopped_reason"] == "startup"
    assert not _locked(run_dir)


# ── the event sink the runtime / session takes ────────────────────────────────────────────


def test_on_event_is_none_on_a_quiet_run(tmp_path):
    """No console and no recorder ⇒ ``None``, so the loops fall back to their own logger sinks."""
    settings = _settings(tmp_path)
    with open_segment(_inputs(settings), command="run", argv=["run"], clock=FakeClock()) as seg:
        assert seg.on_event is None
        seg.finish("time_limit", outcome=_result())


def test_on_event_is_the_level_aware_console_under_v(tmp_path):
    settings = _settings(tmp_path)
    with open_segment(
        _inputs(settings), command="run", argv=["run", "-v"], verbose=1, clock=FakeClock()
    ) as seg:
        assert isinstance(seg.on_event, Console)
        assert seg.on_event.verbose == 1
        seg.finish("time_limit", outcome=_result())


def test_on_event_tees_into_the_recorder_even_when_the_console_is_absent(tmp_path, monkeypatch):
    """A quiet ``--debug`` run records every event silently, which is why the tee is built with a
    ``None`` primary rather than skipped."""
    _llm(monkeypatch, ok=True)
    settings = _settings(tmp_path)

    with open_segment(
        _inputs(settings), command="run", argv=["run", "--debug"], debug=True, clock=FakeClock()
    ) as seg:
        assert isinstance(seg.on_event, EventTee)
        seg.finish("time_limit", outcome=_result())


# ── the band lives in the composition root, so it cannot reach for Typer ──────────────────


def test_the_composition_root_never_imports_typer():
    """The band must not import Typer and must never exit the process: it raises typed errors and
    lets the command map them to red text and an exit code."""
    import noctis.bootstrap as bootstrap

    tree = ast.parse(Path(bootstrap.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "typer" not in imported
    assert "click" not in imported
