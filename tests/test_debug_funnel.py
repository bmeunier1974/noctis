"""The pure funnel ledger + renderers (story #41, epic #36).

Everything under test here is a *pure function over a hand-built event list*: no disk, no clock,
no live run. The behaviours asserted are external — the rendered markdown strings and the JSON
dict — never the ledger's private state. Each event is a caller-stamped :class:`StampedEvent`
(``Event`` carries no timestamp by design; the recorder arrival-stamps it), so the tests inject
plain ``datetime`` literals and read the numbers straight off the output.

The event shapes mirror the real emit sites verified in the story brief:

* tool events (``noctis.research.agent._tool_event``) always carry ``meta["ok"]``,
  ``meta["tool"]`` (the schema name) and ``meta["args"]`` (which holds ``name`` — the strategy —
  for strategy tools); a success merges the gate-facing brief (``promoted``/``rationale``/
  ``n_trials``/``n_failed``/``status``/…), an error leaves ``text`` ending ``ERROR: <full text>``.
* phase events (``noctis.engine.runtime._on_phase_enter``) carry ``meta["phase"]`` = the
  ``Phase`` value (``RESEARCH``/``TRADING``/``CLOSE``/``STOPPED``).
"""

from __future__ import annotations

import json
from datetime import datetime

from noctis.observability import Event
from noctis.observability.debug import (
    StampedEvent,
    render_counts_json,
    render_counts_markdown,
    render_errors_markdown,
    render_summary_markdown,
)

# ── event builders (mirror the real emit sites) ──────────────────────────────────────────────


def _t(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 7, 20, 14, minute, second)


def _tool(
    t: datetime,
    tool: str,
    *,
    ok: bool = True,
    name: str | None = None,
    reason: str | None = None,
    text: str | None = None,
    **brief: object,
) -> StampedEvent:
    """A ``tool`` event exactly as ``_tool_event`` builds it: ``ok``/``tool``/``args`` always,
    the brief merged only on success, ``reason`` (a reject arg) living inside ``args``."""
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
    return StampedEvent(t, Event("tool", text, meta=meta, level=1))


def _phase(t: datetime, phase: str) -> StampedEvent:
    return StampedEvent(
        t, Event("phase", f"{phase} · cycle 0", meta={"phase": phase, "cycle": 0}, level=1)
    )


WINDOW_START = _t(0)
WINDOW_END = datetime(2026, 7, 20, 15, 0, 0)


# ── AC1: the full happy path — exact funnel counts, rows, durations ───────────────────────────


def _happy_events() -> list[StampedEvent]:
    return [
        _phase(_t(0), "RESEARCH"),
        _tool(_t(0, 10), "write_strategy", name="alpha"),
        _tool(_t(0, 20), "run_backtest", name="alpha"),
        _tool(_t(0, 30), "run_sweep", name="alpha", n_trials=40, n_failed=3),
        _tool(
            _t(0, 40),
            "evaluate_vs_champion",
            name="alpha",
            promoted=True,
            rationale="promoted: test metric 1.50 clears the bar into a free slot",
        ),
        _tool(_t(1, 0), "write_strategy", name="beta"),
        _tool(_t(1, 10), "run_backtest", name="beta"),
        _tool(_t(1, 20), "run_sweep", name="beta", n_trials=20, n_failed=0),
        _tool(
            _t(1, 30),
            "evaluate_vs_champion",
            name="beta",
            promoted=False,
            rationale="rejected: does not beat weakest champion",
        ),
        _tool(_t(1, 40), "reject_strategy", name="beta", reason="thesis did not survive holdout"),
    ]


def test_happy_path_funnel_counts_are_exact():
    md = render_counts_markdown(_happy_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert "| write attempts | 2 |" in md
    assert "| written | 2 |" in md
    assert "| backtested | 2 |" in md
    assert "| swept | 2 |" in md
    assert "| compared | 2 |" in md
    assert "| champion | 1 |" in md
    assert "| rejected | 1 |" in md
    assert "| rejected pre-sweep | 0 |" in md


def test_happy_path_per_strategy_rows_carry_the_full_fate():
    md = render_counts_markdown(_happy_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    alpha = _row_for(md, "alpha")
    assert "champion" in alpha
    assert "40 (3)" in alpha  # sweep trials (of which failed)
    assert "promoted: test metric 1.50 clears the bar into a free slot" in alpha
    beta = _row_for(md, "beta")
    assert "rejected" in beta
    assert "20 (0)" in beta
    assert "thesis did not survive holdout" in beta


def test_happy_path_window_header_is_utc():
    md = render_counts_markdown(_happy_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert "2026-07-20 14:00–15:00 UTC" in md


def test_happy_path_json_mirrors_the_markdown():
    doc = render_counts_json(_happy_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["funnel"] == {
        "write_attempts": 2,
        "written": 2,
        "backtested": 2,
        "swept": 2,
        "compared": 2,
        "champion": 1,
        "rejected": 1,
        "rejected_pre_sweep": 0,
    }
    # first-seen order is deterministic
    assert [f["name"] for f in doc["fates"]] == ["alpha", "beta"]
    alpha = doc["fates"][0]
    assert alpha["outcome"] == "champion"
    assert alpha["sweep_trials"] == 40
    assert alpha["sweep_failed"] == 3
    beta = doc["fates"][1]
    assert beta["outcome"] == "rejected"
    assert beta["reason"] == "thesis did not survive holdout"
    # json-serialisable end to end
    json.dumps(doc)


# ── AC2: rejected before any sweep is counted distinctly and visible in the row ───────────────


def _reject_events() -> list[StampedEvent]:
    return [
        _tool(_t(0, 10), "write_strategy", name="gamma"),
        _tool(_t(0, 20), "run_backtest", name="gamma"),
        _tool(_t(0, 30), "reject_strategy", name="gamma", reason="no edge before tuning"),
        _tool(_t(1, 0), "write_strategy", name="delta"),
        _tool(_t(1, 10), "run_sweep", name="delta", n_trials=10, n_failed=1),
        _tool(_t(1, 20), "reject_strategy", name="delta", reason="swept then killed"),
    ]


def test_rejected_before_any_sweep_is_a_distinct_count():
    doc = render_counts_json(_reject_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["funnel"]["rejected"] == 2
    assert doc["funnel"]["rejected_pre_sweep"] == 1  # only gamma


def test_rejected_before_any_sweep_shows_in_its_row():
    md = render_counts_markdown(_reject_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert "| rejected pre-sweep | 1 |" in md
    assert "rejected pre-sweep" in _row_for(md, "gamma")
    # delta was swept first — plain "rejected", never flagged pre-sweep
    delta = _row_for(md, "delta")
    assert "rejected" in delta
    assert "pre-sweep" not in delta


# ── AC3: a write-fail-only stream renders correctly ───────────────────────────────────────────


def _write_fail_events() -> list[StampedEvent]:
    return [
        _tool(_t(0, 10), "write_strategy", ok=False, name="epsilon"),
        _tool(_t(0, 20), "write_strategy", ok=False, name="epsilon"),
    ]


def test_write_fail_only_stream_counts_attempts_but_no_writes():
    md = render_counts_markdown(
        _write_fail_events(), window_start=WINDOW_START, window_end=WINDOW_END
    )
    assert "| write attempts | 2 |" in md
    assert "| written | 0 |" in md
    assert "| champion | 0 |" in md
    row = _row_for(md, "epsilon")
    assert "write failed" in row


def test_write_fail_only_stream_json():
    doc = render_counts_json(_write_fail_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["funnel"]["write_attempts"] == 2
    assert doc["funnel"]["written"] == 0
    fate = doc["fates"][0]
    assert fate["name"] == "epsilon"
    assert fate["write_attempts"] == 2
    assert fate["writes"] == 0
    assert fate["outcome"] == "write failed"


# ── AC4: the errors renderer reproduces full, untruncated failure text ────────────────────────


def test_errors_renderer_reproduces_full_multiline_text():
    boom = (
        'run_sweep({"name": "omega"}) -> ERROR: worker pool crashed mid-sweep\n'
        "Traceback (most recent call last):\n"
        '  File "sweep.py", line 412, in run_sweep\n'
        "    raise RuntimeError('the pool wedged and was force-killed')\n"
        "RuntimeError: the pool wedged and was force-killed"
    )
    events = [_tool(_t(5), "run_sweep", ok=False, name="omega", text=boom)]
    md = render_errors_markdown(events, window_start=WINDOW_START, window_end=WINDOW_END)
    # full text, verbatim, newlines preserved (never collapsed like render_plain does)
    assert boom in md
    assert "Traceback (most recent call last):" in md
    assert "RuntimeError: the pool wedged and was force-killed" in md
    assert "\n" in md


def test_errors_renderer_lists_every_failure_and_a_count():
    events = [
        _tool(_t(1), "write_strategy", ok=False, name="a", text="write_strategy(...) -> ERROR: A"),
        _tool(_t(2), "run_backtest", ok=True, name="a"),  # success — not an error
        _tool(_t(3), "run_sweep", ok=False, name="a", text="run_sweep(...) -> ERROR: B"),
    ]
    md = render_errors_markdown(events, window_start=WINDOW_START, window_end=WINDOW_END)
    assert "2 failure" in md
    assert "ERROR: A" in md
    assert "ERROR: B" in md


def test_errors_renderer_with_no_failures_is_sane():
    events = [_tool(_t(1), "run_backtest", ok=True, name="a")]
    md = render_errors_markdown(events, window_start=WINDOW_START, window_end=WINDOW_END)
    assert "no failures" in md.lower()


# ── AC5: legacy-loop sessions render the honesty line, never a zero-filled funnel ─────────────

_LEGACY_LINE = (
    "research loop: legacy (proposer/Optuna) — funnel not instrumented; "
    "counts below cover phase timing only"
)


def test_legacy_loop_renders_the_exact_honesty_line():
    md = render_counts_markdown(
        _happy_events(),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        funnel_instrumented=False,
    )
    assert _LEGACY_LINE in md


def test_legacy_loop_omits_the_funnel_table_and_rows():
    md = render_counts_markdown(
        _happy_events(),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        funnel_instrumented=False,
    )
    assert "## Funnel" not in md
    assert "| written |" not in md
    assert "alpha" not in md  # no per-strategy rows leak
    # phase timing survives — that is what the honesty line says the counts still cover
    assert "## Phase timing" in md


def test_legacy_loop_json_has_no_funnel():
    doc = render_counts_json(
        _happy_events(),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        funnel_instrumented=False,
    )
    assert doc["funnel_instrumented"] is False
    assert doc["funnel"] is None
    assert doc["fates"] == []
    assert "phase_seconds" in doc  # timing still reported


# ── AC (phase timing): exact seconds from hand-stamped phase events ───────────────────────────


def _phase_events() -> list[StampedEvent]:
    # a 10-minute idle lead-in (window opens at 14:00, first phase at 14:10), then
    # RESEARCH 30m, TRADING 15m, CLOSE runs to the window end (5m).
    return [
        _phase(_t(10), "RESEARCH"),
        _phase(_t(40), "TRADING"),
        _phase(_t(55), "CLOSE"),
    ]


def test_phase_durations_are_exact_in_json():
    doc = render_counts_json(_phase_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["phase_seconds"] == {
        "research": 1800.0,
        "trading": 900.0,
        "close": 300.0,
        "idle_wait": 600.0,
    }


def test_phase_durations_render_readable_in_markdown():
    md = render_counts_markdown(_phase_events(), window_start=WINDOW_START, window_end=WINDOW_END)
    assert "| research | 0:30:00 |" in md
    assert "| trading | 0:15:00 |" in md
    assert "| close | 0:05:00 |" in md
    assert "| idle-wait | 0:10:00 |" in md


def test_idle_wait_is_the_whole_window_when_no_phase_events():
    doc = render_counts_json([], window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["phase_seconds"]["idle_wait"] == 3600.0
    assert doc["phase_seconds"]["research"] == 0.0


# ── empty stream renders something sane, no crash ─────────────────────────────────────────────


def test_empty_stream_renders_without_crashing():
    md = render_counts_markdown([], window_start=WINDOW_START, window_end=WINDOW_END)
    assert "| written | 0 |" in md
    assert "| champion | 0 |" in md
    doc = render_counts_json([], window_start=WINDOW_START, window_end=WINDOW_END)
    assert doc["fates"] == []
    assert doc["funnel"]["written"] == 0
    json.dumps(doc)
    errors = render_errors_markdown([], window_start=WINDOW_START, window_end=WINDOW_END)
    assert "no failures" in errors.lower()


# ── summary: cumulative whole-run rollup over ALL events ──────────────────────────────────────


def test_summary_rolls_up_all_events_cumulatively():
    events = _happy_events() + _reject_events()
    md = render_summary_markdown(events, window_start=WINDOW_START, window_end=WINDOW_END)
    assert "# QA summary" in md
    # alpha promoted; beta + gamma + delta rejected → 1 champion, 3 rejected
    assert "| champion | 1 |" in md
    assert "| rejected | 3 |" in md
    assert "| rejected pre-sweep | 1 |" in md  # only gamma


def test_summary_accepts_optional_notes():
    md = render_summary_markdown(
        [],
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        notes=["self-disabled: budget exhausted"],
    )
    assert "self-disabled: budget exhausted" in md


def test_summary_honours_the_legacy_honesty_line():
    md = render_summary_markdown(
        _happy_events(),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        funnel_instrumented=False,
    )
    assert _LEGACY_LINE in md
    assert "## Funnel" not in md


# ── helpers ──────────────────────────────────────────────────────────────────────────────────


def _row_for(md: str, name: str) -> str:
    """The per-strategy table row whose first cell is ``name`` — the row-scoped assertion seam."""
    for line in md.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] == name:
            return line
    raise AssertionError(f"no per-strategy row for {name!r} in:\n{md}")
