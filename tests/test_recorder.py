"""The recorder — the one disk-touching module in the debug subpackage (story #43, epic #36).

Every behaviour asserted here is *external and on-disk*: which segment folders exist, what the
``events.jsonl`` lines parse to, the numbers in each ``counts.json``, the manifest fields. Never a
write-call sequence and never private state — the recorder is a black box driven by an injected
clock, exactly as the CLI (story #45) will drive it and as the epic's contract prescribes.

The clock is a mutable holder the test advances (``FakeClock``): construction reads it once (the
``started`` stamp), each ``__call__`` reads it once (the arrival stamp), ``close()`` reads it once
(the ``stopped`` stamp). Setting ``clock.now`` before each drive is not brittle the way a
pop-from-a-list clock would be — the test names the instant each event arrives.

Event builders mirror the real emit sites (see ``tests/test_debug_funnel.py``): ``tool`` events
carry ``meta["ok"]``/``meta["tool"]``/``meta["args"]``; ``phase`` events carry ``meta["phase"]``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from noctis.observability import Event
from noctis.observability.debug import Recorder

# ── the injected clock: a mutable holder the test advances ────────────────────────────────────


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock reads reach the recorder."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def at(self, dt: datetime) -> FakeClock:
        self.now = dt
        return self


# ── event builders (mirror the real emit sites, as in test_debug_funnel) ──────────────────────


def _tool(
    tool: str,
    *,
    ok: bool = True,
    name: str | None = None,
    reason: str | None = None,
    text: str | None = None,
    **brief: object,
) -> Event:
    args: dict = {}
    if name is not None:
        args["name"] = name
    if reason is not None:
        args["reason"] = reason
    meta: dict = {"ok": ok, "tool": tool, "args": args}
    if ok:
        meta.update(brief)
    if text is None:
        text = f"{tool}(...) -> {'ok' if ok else 'ERROR: something went wrong'}"
    return Event("tool", text, meta=meta, level=1)


def _phase(phase: str) -> Event:
    return Event("phase", f"{phase} · cycle 0", meta={"phase": phase, "cycle": 0}, level=1)


def _at(hour: int, minute: int = 0, second: int = 0, micro: int = 0) -> datetime:
    return datetime(2026, 7, 20, hour, minute, second, micro)


START = _at(14, 0, 0)

_MANIFEST = {
    "argv": ["noctis", "run", "--debug"],
    "mode": "paper",
    "config_digest": "sha256:abc123",
    "versions": {"noctis": "0.1.0", "python": "3.11"},
}

_RUN_ID = "20260720T140000Z-a3f9c1"

_T_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _make(tmp_path: Path, clock: FakeClock) -> Recorder:
    return Recorder(tmp_path, run_id=_RUN_ID, clock=clock, manifest=dict(_MANIFEST))


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / _RUN_ID


def _jsonl(seg: Path) -> list[dict]:
    lines = (seg / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── AC1: an hour boundary → two segments, per-hour counters reset, summary cumulative ─────────


def test_hour_boundary_makes_two_segments_with_reset_counters_and_cumulative_summary(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    # hour 0: alpha is written, backtested, swept, promoted.
    def drive(dt: datetime, ev: Event) -> None:
        clock.at(dt)
        rec(ev)

    drive(_at(14, 10), _tool("write_strategy", name="alpha"))
    drive(_at(14, 20), _tool("run_backtest", name="alpha"))
    drive(_at(14, 30), _tool("run_sweep", name="alpha", n_trials=40, n_failed=3))
    drive(
        _at(14, 40),
        _tool("evaluate_vs_champion", name="alpha", promoted=True, rationale="clears the bar"),
    )
    # hour 1: beta is written and backtested (crosses the boundary → rollover).
    drive(_at(15, 5), _tool("write_strategy", name="beta"))
    drive(_at(15, 15), _tool("run_backtest", name="beta"))
    clock.at(_at(15, 30))
    rec.close()

    run = _run_dir(tmp_path)
    assert (run / "h00").is_dir()
    assert (run / "h01").is_dir()

    h0 = json.loads((run / "h00" / "counts.json").read_text())
    h1 = json.loads((run / "h01" / "counts.json").read_text())
    # per-hour counters reset: h0 sees only alpha, h1 only beta
    assert h0["funnel"]["written"] == 1
    assert h0["funnel"]["champion"] == 1
    assert [f["name"] for f in h0["fates"]] == ["alpha"]
    assert h1["funnel"]["written"] == 1
    assert h1["funnel"]["champion"] == 0
    assert [f["name"] for f in h1["fates"]] == ["beta"]

    # summary holds running totals across BOTH hours
    summary = (run / "summary.md").read_text()
    assert "# QA summary" in summary
    assert "| written | 2 |" in summary
    assert "| champion | 1 |" in summary


def test_hour_boundary_counts_markdown_are_per_segment(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    def drive(dt: datetime, ev: Event) -> None:
        clock.at(dt)
        rec(ev)

    drive(_at(14, 10), _tool("write_strategy", name="alpha"))
    drive(_at(15, 5), _tool("write_strategy", name="beta"))
    clock.at(_at(15, 30))
    rec.close()

    run = _run_dir(tmp_path)
    h0_md = (run / "h00" / "counts.md").read_text()
    h1_md = (run / "h01" / "counts.md").read_text()
    # each hour's window header is UTC and covers only that hour
    assert "2026-07-20 14:00–15:00 UTC" in h0_md
    assert "2026-07-20 15:00–16:00 UTC" in h1_md
    # the strategy of the other hour never leaks into this hour's rows
    assert "beta" not in h0_md
    assert "alpha" not in h1_md


# ── AC2: idle hours write no empty folders; the manifest duration stays authoritative ─────────


def test_idle_hours_produce_no_empty_folders_and_duration_is_authoritative(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    def drive(dt: datetime, ev: Event) -> None:
        clock.at(dt)
        rec(ev)

    drive(_at(14, 10), _tool("write_strategy", name="alpha"))  # hour 0
    # two idle hours (h01, h02) then an event three hours in → hour 3
    drive(_at(17, 20), _tool("write_strategy", name="beta"))  # hour 3
    clock.at(_at(17, 30))
    rec.close()

    run = _run_dir(tmp_path)
    assert (run / "h00").is_dir()
    assert (run / "h03").is_dir()
    assert not (run / "h01").exists()
    assert not (run / "h02").exists()
    # only the two touched segments exist — no empty idle folders
    segments = sorted(p.name for p in run.iterdir() if p.is_dir())
    assert segments == ["h00", "h03"]

    manifest = json.loads((run / "run.json").read_text())
    # duration is clock time end-to-end (14:00 → 17:30), authoritative even across idle hours
    assert manifest["duration_s"] == 12600.0


# ── AC3: events.jsonl lines are arrival-stamped and match the documented shape ────────────────


def test_events_jsonl_lines_are_arrival_stamped_and_match_shape(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    clock.at(_at(14, 51, 2, 418000))
    rec(_phase("RESEARCH"))
    clock.at(_at(14, 51, 2, 418000))
    rec(_tool("run_sweep", name="omega", n_trials=40, n_failed=3, text="run_sweep({...}) -> n=40"))
    clock.at(_at(15, 0, 0))
    rec.close()

    lines = _jsonl(_run_dir(tmp_path) / "h00")
    assert len(lines) == 2

    phase_line = lines[0]
    assert _T_RE.match(phase_line["t"])
    assert phase_line["kind"] == "phase"
    assert phase_line["phase"] == "RESEARCH"
    assert phase_line["text"]  # always present
    assert phase_line["meta"] == {"phase": "RESEARCH", "cycle": 0}
    # a phase frame carries no tool/ok
    assert "tool" not in phase_line
    assert "ok" not in phase_line

    tool_line = lines[1]
    assert _T_RE.match(tool_line["t"])
    assert tool_line["t"] == "2026-07-20T14:51:02.418Z"
    assert isinstance(tool_line["el"], float)
    assert round(tool_line["el"], 1) == tool_line["el"]  # one decimal
    assert tool_line["el"] == 3062.4  # 51m02.418s after 14:00
    assert tool_line["phase"] == "RESEARCH"  # tracked from the phase frame
    assert tool_line["kind"] == "tool"
    assert tool_line["tool"] == "run_sweep"  # lifted from meta
    assert tool_line["ok"] is True  # lifted from meta
    assert tool_line["text"] == "run_sweep({...}) -> n=40"
    assert tool_line["meta"]["n_trials"] == 40
    assert tool_line["meta"]["n_failed"] == 3


def test_phase_is_null_before_any_phase_event(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 5))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 30))
    rec.close()

    lines = _jsonl(_run_dir(tmp_path) / "h00")
    assert lines[0]["phase"] is None  # honest placeholder before any phase frame


def test_str_event_is_recorded_as_a_feed_line(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 5))
    rec("web_search auto-disabled: no key")  # a bare legacy pre-formatted str line
    clock.at(_at(14, 30))
    rec.close()

    lines = _jsonl(_run_dir(tmp_path) / "h00")
    assert len(lines) == 1
    line = lines[0]
    assert line["kind"] == "feed"
    assert line["text"] == "web_search auto-disabled: no key"
    assert line["meta"] == {}
    assert "tool" not in line
    assert _T_RE.match(line["t"])


# ── AC4: summary rewritten at rollover and close; run.json carries the manifest fields ────────


def test_run_json_written_at_construction_with_null_stopped(tmp_path):
    clock = FakeClock(START)
    _make(tmp_path, clock)  # construction alone writes the manifest

    manifest = json.loads((_run_dir(tmp_path) / "run.json").read_text())
    assert manifest["run_id"] == _RUN_ID
    assert manifest["argv"] == ["noctis", "run", "--debug"]
    assert manifest["mode"] == "paper"
    assert manifest["config_digest"] == "sha256:abc123"
    assert manifest["versions"] == {"noctis": "0.1.0", "python": "3.11"}
    assert _T_RE.match(manifest["started"])
    assert manifest["stopped"] is None
    assert manifest["duration_s"] is None


def test_close_stamps_stopped_and_duration(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 5))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 45))
    rec.close()

    manifest = json.loads((_run_dir(tmp_path) / "run.json").read_text())
    assert _T_RE.match(manifest["stopped"])
    assert manifest["duration_s"] == 2700.0  # 45 minutes


def test_summary_is_written_at_rollover_before_close(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    def drive(dt: datetime, ev: Event) -> None:
        clock.at(dt)
        rec(ev)

    drive(_at(14, 10), _tool("write_strategy", name="alpha"))
    assert not (_run_dir(tmp_path) / "summary.md").exists()  # no rollover yet
    drive(_at(15, 5), _tool("write_strategy", name="beta"))  # rollover → summary written
    assert (_run_dir(tmp_path) / "summary.md").exists()


def test_summary_rewritten_at_close_reflects_all_events(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 10))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 20))
    rec(_tool("reject_strategy", name="alpha", reason="no edge"))
    clock.at(_at(14, 45))
    rec.close()

    summary = (_run_dir(tmp_path) / "summary.md").read_text()
    assert "| written | 1 |" in summary
    assert "| rejected | 1 |" in summary


# ── AC5: no background thread, async IO, or per-event fsync anywhere in the module ────────────


def test_driving_the_recorder_spawns_no_threads(tmp_path):
    baseline = threading.active_count()
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    for i in range(20):
        clock.at(_at(14, i))
        rec(_tool("write_strategy", name=f"s{i}"))
    clock.at(_at(15, 30))
    rec.close()
    assert threading.active_count() == baseline


def test_module_source_uses_no_threads_asyncio_or_fsync():
    import noctis.observability.debug.recorder as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    # strip the module docstring, which references the anti-thread history by name on purpose
    body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
    assert "threading" not in body
    assert "import asyncio" not in body
    assert "asyncio" not in body
    assert "fsync" not in body
    assert "Thread(" not in body


# ── close is idempotent (cheap now, makes #44/#45 easier) ─────────────────────────────────────


def test_close_is_idempotent(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 10))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 45))
    rec.close()
    first = json.loads((_run_dir(tmp_path) / "run.json").read_text())

    clock.at(_at(20, 0))  # a much later time — a second close must NOT re-stamp
    rec.close()  # harmless
    second = json.loads((_run_dir(tmp_path) / "run.json").read_text())
    assert first == second  # stopped/duration frozen at the first close


# ── phase events flush the buffered jsonl to disk (a phase transition is a flush point) ───────


def test_phase_event_flushes_jsonl_without_close(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 5))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 10))
    rec(_phase("RESEARCH"))  # phase transition → flush

    # events.jsonl is on disk with both lines, without any close having run
    lines = _jsonl(_run_dir(tmp_path) / "h00")
    assert len(lines) == 2
    assert lines[0]["text"].startswith("write_strategy")
    assert lines[1]["kind"] == "phase"


# ── mark_legacy_research toggles the honesty line into subsequent renders ─────────────────────


def test_mark_legacy_research_suppresses_the_funnel_in_renders(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    rec.mark_legacy_research()
    clock.at(_at(14, 10))
    rec(_tool("write_strategy", name="alpha"))
    clock.at(_at(14, 45))
    rec.close()

    run = _run_dir(tmp_path)
    counts_md = (run / "h00" / "counts.md").read_text()
    counts_json = json.loads((run / "h00" / "counts.json").read_text())
    summary = (run / "summary.md").read_text()
    assert "funnel not instrumented" in counts_md
    assert "## Funnel" not in counts_md
    assert counts_json["funnel_instrumented"] is False
    assert counts_json["funnel"] is None
    assert "funnel not instrumented" in summary


# ── story #45 surface: run_id / run_dir properties + the compact funnel one-liner ─────────────


def test_run_id_and_run_dir_are_read_only_properties(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    assert rec.run_id == _RUN_ID
    assert rec.run_dir == _run_dir(tmp_path)


def test_funnel_line_reports_compact_whole_run_counts(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)

    def drive(dt: datetime, ev: Event) -> None:
        clock.at(dt)
        rec(ev)

    drive(_at(14, 10), _tool("write_strategy", name="alpha"))
    drive(_at(14, 20), _tool("run_backtest", name="alpha"))
    drive(_at(14, 30), _tool("run_sweep", name="alpha", n_trials=40, n_failed=3))
    drive(
        _at(14, 40),
        _tool("evaluate_vs_champion", name="alpha", promoted=True, rationale="clears the bar"),
    )
    drive(_at(14, 50), _tool("write_strategy", name="beta"))
    drive(_at(14, 55), _tool("reject_strategy", name="beta", reason="no edge"))

    line = rec.funnel_line()
    assert line == "written=2 backtested=1 swept=1 compared=1 champions=1 rejected=1"


def test_funnel_line_is_available_before_close(tmp_path):
    """The stop echo reads it before or after close — a still-open recorder answers honestly."""
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(14, 10))
    rec(_tool("write_strategy", name="alpha"))
    assert rec.funnel_line() == "written=1 backtested=0 swept=0 compared=0 champions=0 rejected=0"


def test_funnel_line_says_legacy_when_marked(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    rec.mark_legacy_research()
    clock.at(_at(14, 10))
    rec(_tool("write_strategy", name="alpha"))
    line = rec.funnel_line()
    assert "legacy" in line.lower()
    assert "written=" not in line  # no fake zero-filled funnel for an uninstrumented loop


def test_funnel_line_is_honest_when_the_latch_has_tripped(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    _patch_write_raises_on(monkeypatch, run_dir, 1)  # trip on construction's run.json write
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    assert rec.disabled is True
    line = rec.funnel_line()
    assert "written=" not in line  # never a comforting all-zeros funnel after a failure
    assert "disabled" in line.lower() or "unavailable" in line.lower()


# ── errors document carries this hour's failures, untruncated ─────────────────────────────────


def test_errors_md_carries_this_hours_failures(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    boom = "run_sweep(...) -> ERROR: worker pool wedged\nTraceback: ...\nRuntimeError: force-killed"
    clock.at(_at(14, 10))
    rec(_tool("run_sweep", ok=False, name="omega", text=boom))
    clock.at(_at(14, 45))
    rec.close()

    errors = (_run_dir(tmp_path) / "h00" / "errors.md").read_text()
    assert boom in errors  # full, untruncated text


# ── an entirely idle run still writes a manifest and a summary, no segment folders ────────────


def test_idle_run_writes_manifest_and_summary_but_no_segments(tmp_path):
    clock = FakeClock(START)
    rec = _make(tmp_path, clock)
    clock.at(_at(16, 0))
    rec.close()

    run = _run_dir(tmp_path)
    assert (run / "run.json").exists()
    assert (run / "summary.md").exists()
    assert not any(p.is_dir() and p.name.startswith("h") for p in run.iterdir())
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["duration_s"] == 7200.0  # two hours of idle, still authoritative


# ── story #44: the fail-safe latch — disable on the first internal failure ────────────────────
#
# Every behaviour asserted here is still external: `caplog` records, the `disabled` property, and
# the on-disk summary. The one seam these tests reach into is the *filesystem* — `Path.write_text`,
# the exact call the recorder writes through — patched to raise on a chosen recorder write. That is
# the narrowest honest way to inject an internal failure without asserting private state.

RECORDER_LOGGER = "noctis.observability.debug.recorder"


def _patch_write_raises_on(monkeypatch, run_dir: Path, n: int) -> dict:
    """Make the recorder's own disk writes raise once, on the *n*-th write into its run tree.

    Only writes under ``run_dir`` are counted — the recorder is the sole writer there, so pytest's
    own I/O never perturbs the count. Every other write delegates to the real implementation.
    """
    real = Path.write_text
    state = {"n": 0}

    def counting(self: Path, *args: object, **kwargs: object):
        if str(self).startswith(str(run_dir)):
            state["n"] += 1
            if state["n"] == n:
                raise OSError("simulated disk failure on the recorder's write")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", counting)
    return state


def _recorder_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == RECORDER_LOGGER and r.levelno == logging.WARNING]


# AC1: writer raises on the 3rd call → the run completes, the recorder latches off, exactly one
# warning is logged, and no exception escapes into the caller.
def test_writer_raising_on_third_call_latches_without_escaping(tmp_path, monkeypatch, caplog):
    run_dir = _run_dir(tmp_path)
    _patch_write_raises_on(monkeypatch, run_dir, 3)
    clock = FakeClock(START)

    with caplog.at_level(logging.WARNING, logger=RECORDER_LOGGER):
        rec = _make(tmp_path, clock)  # write #1: run.json

        def drive(dt: datetime, ev: Event) -> None:
            clock.at(dt)
            rec(ev)  # must never raise, even when the writer fails underneath

        drive(_at(14, 10), _phase("RESEARCH"))  # write #2: h00/events.jsonl (phase flush)
        drive(_at(15, 5), _tool("write_strategy", name="beta"))  # rollover → write #3 raises
        # the run continues past the failure — later calls are silent no-ops, not retries
        drive(_at(15, 15), _tool("run_backtest", name="beta"))
        clock.at(_at(15, 30))
        rec.close()  # a no-op now; must not raise and must not re-warn

    assert rec.disabled is True  # AC1: the recorder is disabled
    warnings = _recorder_warnings(caplog)
    assert len(warnings) == 1  # AC1: exactly one warning
    assert _RUN_ID in warnings[0].getMessage()  # names the run

    # AC3: the final summary states the disablement and names the hour coverage stopped
    summary = (run_dir / "summary.md").read_text()
    assert "self-disabled" in summary
    assert "h00" in summary  # coverage stopped in the open segment (h00) at trip time


# AC2: after the latch trips, every public method is a no-op — no retries, no further warnings,
# no further disk writes, no exceptions.
def test_every_public_method_is_a_noop_after_the_latch_trips(tmp_path, monkeypatch, caplog):
    run_dir = _run_dir(tmp_path)
    state = _patch_write_raises_on(monkeypatch, run_dir, 1)  # trip on construction's run.json
    clock = FakeClock(START)

    with caplog.at_level(logging.WARNING, logger=RECORDER_LOGGER):
        rec = _make(tmp_path, clock)  # write #1 raises → latch; write #2 = best-effort summary
        assert rec.disabled is True
        writes_after_trip = state["n"]

        # exercise the whole public surface — none may raise, write, or warn again
        clock.at(_at(14, 10))
        rec(_tool("write_strategy", name="alpha"))
        rec(_phase("RESEARCH"))
        rec("a bare legacy line")
        rec.flush()
        rec.mark_legacy_research()
        clock.at(_at(15, 0))
        rec.close()

    assert state["n"] == writes_after_trip  # no method touched disk after the latch
    assert len(_recorder_warnings(caplog)) == 1  # exactly one warning — no retries, no spam


# The latch can trip before any segment opens (a construction-time write failure): the object
# still constructs, and the honesty note names the pre-event coverage boundary.
def test_trip_before_first_segment_still_constructs_and_notes_pre_event(
    tmp_path, monkeypatch, caplog
):
    run_dir = _run_dir(tmp_path)
    _patch_write_raises_on(monkeypatch, run_dir, 1)  # construction's run.json write raises
    clock = FakeClock(START)

    with caplog.at_level(logging.WARNING, logger=RECORDER_LOGGER):
        rec = _make(tmp_path, clock)  # must NOT raise into the caller (the engine)

    assert rec.disabled is True
    assert len(_recorder_warnings(caplog)) == 1
    summary = (run_dir / "summary.md").read_text()
    assert "self-disabled" in summary
    assert "before the first event" in summary  # honest: nothing was ever covered


# The latch can also trip inside close(): close must swallow it, latch, and still note the hour.
def test_trip_during_close_latches_and_notes_the_hour(tmp_path, monkeypatch, caplog):
    run_dir = _run_dir(tmp_path)
    clock = FakeClock(START)

    with caplog.at_level(logging.WARNING, logger=RECORDER_LOGGER):
        rec = _make(tmp_path, clock)  # write: run.json (unpatched)
        clock.at(_at(14, 10))
        rec(_tool("write_strategy", name="alpha"))  # buffered, no write yet
        _patch_write_raises_on(monkeypatch, run_dir, 1)  # next recorder write fails
        clock.at(_at(14, 45))
        rec.close()  # _finalize_segment's first write raises → must not escape

    assert rec.disabled is True
    assert len(_recorder_warnings(caplog)) == 1
    summary = (run_dir / "summary.md").read_text()
    assert "self-disabled" in summary
    assert "h00" in summary
