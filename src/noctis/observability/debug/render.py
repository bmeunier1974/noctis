"""Pure renderers: a :class:`~noctis.observability.debug.funnel.Ledger` + events → QA documents.

Four documents, all pure functions of the (stamped) event list and the segment window — no disk,
no clock, no I/O. The ledger is derived from the same events (a renderer is a pure function of
``ledger(events)`` and ``events``), so a caller hands over one list and a window and gets back a
string or a JSON-ready dict. The recorder (story #43) owns the clock and the disk and simply
feeds these.

* **counts markdown** — the funnel table, per-strategy fate rows, phase-time accounting, under a
  UTC window header.
* **counts JSON** — the same data, machine-readable and ``json.dumps``-able, with deterministic
  (first-seen) fate ordering.
* **errors markdown** — every failure-shaped event with its *full, untruncated* text.
* **summary markdown** — the cumulative whole-run rollup, same shape as counts.

**The honesty contract (AGENTS.md rule 2, in spirit).** The legacy proposer/Optuna research loop
is not funnel-instrumented, so a zero-filled funnel table would be a comforting lie — it would
read as "nothing happened" when the truth is "we did not measure." When ``funnel_instrumented``
is False the renderers print exactly :data:`LEGACY_NOTICE` where the funnel would go and emit no
funnel table or per-strategy rows at all; phase timing (which *is* measured either way) stays.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from noctis.observability.debug.funnel import (
    Ledger,
    StampedEvent,
    build_ledger,
    phase_durations,
)

# The exact line a legacy-loop session shows instead of a funnel table. Kept as a module constant
# so the recorder and the tests reference the one canonical string.
LEGACY_NOTICE = (
    "research loop: legacy (proposer/Optuna) — funnel not instrumented; "
    "counts below cover phase timing only"
)

# Markdown label per timing bucket (the internal key differs for idle-wait: json-safe underscore
# in the dict, a readable hyphen in the table).
_PHASE_LABELS: tuple[tuple[str, str], ...] = (
    ("research", "research"),
    ("trading", "trading"),
    ("close", "close"),
    ("idle_wait", "idle-wait"),
)


def _window_header(start: datetime, end: datetime) -> str:
    """``2026-07-20 14:00–15:00 UTC`` — the segment window, UTC by contract. The end date is
    only spelled out when the window crosses midnight, which hour segments never do."""
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC"
    return f"{start:%Y-%m-%d %H:%M}–{end:%Y-%m-%d %H:%M} UTC"


def _fmt_hms(seconds: float) -> str:
    """Whole-second ``H:MM:SS`` — the natural, deterministic shape for phase time in a report."""
    return str(timedelta(seconds=int(round(seconds))))


def _cell(text: str) -> str:
    """Sanitise a value for a markdown table cell: one line, pipes escaped so a rationale that
    contains ``|`` cannot fracture the row. The errors renderer never uses this — failure text is
    reproduced verbatim in a fenced block instead."""
    if not text:
        return ""
    return " ".join(text.split()).replace("|", "\\|")


def _funnel_section(ledger: Ledger) -> list[str]:
    counts = ledger.counts
    lines = ["## Funnel", "", "| stage | count |", "| --- | --- |"]
    for label, value in (
        ("write attempts", counts.write_attempts),
        ("written", counts.written),
        ("backtested", counts.backtested),
        ("swept", counts.swept),
        ("compared", counts.compared),
        ("champion", counts.champion),
        ("rejected", counts.rejected),
        ("rejected pre-sweep", counts.rejected_pre_sweep),
    ):
        lines.append(f"| {label} | {value} |")
    lines += ["", "## Per-strategy fates", ""]
    lines.append(
        "| strategy | write attempts | backtests | sweep trials (failed) "
        "| comparisons | outcome | reason |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    if not ledger.fates:
        lines.append("| _(none)_ |  |  |  |  |  |  |")
    for fate in ledger.fates:
        lines.append(
            f"| {fate.name} | {fate.write_attempts} | {fate.backtests} "
            f"| {fate.sweep_trials} ({fate.sweep_failed}) | {fate.comparisons} "
            f"| {fate.outcome} | {_cell(fate.reason)} |"
        )
    lines.append("")
    return lines


def _phase_section(
    events: Sequence[StampedEvent], window_start: datetime, window_end: datetime
) -> list[str]:
    durations = phase_durations(events, window_start, window_end)
    lines = ["## Phase timing", "", "| phase | duration |", "| --- | --- |"]
    for key, label in _PHASE_LABELS:
        lines.append(f"| {label} | {_fmt_hms(durations[key])} |")
    lines.append("")
    return lines


def _counts_body(
    events: Sequence[StampedEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    funnel_instrumented: bool,
    title: str,
    segment_label: str | None,
    notes: Sequence[str] | None,
) -> str:
    """Shared body for the counts and summary documents — the funnel (or the honesty line) over
    the phase-time accounting, under a UTC window header."""
    heading = f"# {title}"
    if segment_label:
        heading += f" — {segment_label}"
    lines: list[str] = [heading, "", _window_header(window_start, window_end), ""]

    if funnel_instrumented:
        lines += _funnel_section(build_ledger(events))
    else:
        lines += [LEGACY_NOTICE, ""]

    lines += _phase_section(events, window_start, window_end)

    if notes:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in notes]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_counts_markdown(
    events: Sequence[StampedEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    funnel_instrumented: bool = True,
    segment_label: str | None = None,
) -> str:
    """The per-segment counts document: funnel table + per-strategy rows + phase timing."""
    return _counts_body(
        events,
        window_start=window_start,
        window_end=window_end,
        funnel_instrumented=funnel_instrumented,
        title="QA counts",
        segment_label=segment_label,
        notes=None,
    )


def render_counts_json(
    events: Sequence[StampedEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    funnel_instrumented: bool = True,
    segment_label: str | None = None,
) -> dict:
    """The counts document as a ``json.dumps``-able dict — the same numbers, machine-readable.

    ``funnel`` is ``None`` and ``fates`` empty for a legacy (uninstrumented) loop, so a consumer
    can tell "no funnel measured" from "a funnel of all zeros". Fate order is first-seen.
    """
    doc: dict = {
        "segment": segment_label,
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "label": _window_header(window_start, window_end),
        },
        "funnel_instrumented": funnel_instrumented,
        "phase_seconds": phase_durations(events, window_start, window_end),
    }
    if not funnel_instrumented:
        doc["funnel"] = None
        doc["fates"] = []
        return doc

    ledger = build_ledger(events)
    counts = ledger.counts
    doc["funnel"] = {
        "write_attempts": counts.write_attempts,
        "written": counts.written,
        "backtested": counts.backtested,
        "swept": counts.swept,
        "compared": counts.compared,
        "champion": counts.champion,
        "rejected": counts.rejected,
        "rejected_pre_sweep": counts.rejected_pre_sweep,
    }
    doc["fates"] = [
        {
            "name": fate.name,
            "write_attempts": fate.write_attempts,
            "writes": fate.writes,
            "backtests": fate.backtests,
            "sweeps": fate.sweeps,
            "sweep_trials": fate.sweep_trials,
            "sweep_failed": fate.sweep_failed,
            "comparisons": fate.comparisons,
            "promoted": fate.promoted,
            "rejected": fate.rejected,
            "outcome": fate.outcome,
            "reason": fate.reason,
        }
        for fate in ledger.fates
    ]
    return doc


def render_errors_markdown(
    events: Sequence[StampedEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    segment_label: str | None = None,
) -> str:
    """Every failure-shaped event with its *full, untruncated* text.

    A failure is any event whose ``meta["ok"]`` is exactly ``False`` — the flag every tool error
    carries (#37). Refusals and other non-``ok``-flagged events are out of scope by design: they
    are not tool failures. The text is reproduced verbatim inside a fenced block — never
    truncated, never collapsed to one line the way ``render_plain`` does — because a debug run's
    whole value is the full traceback.
    """
    heading = "# QA errors"
    if segment_label:
        heading += f" — {segment_label}"
    lines: list[str] = [heading, "", _window_header(window_start, window_end), ""]

    failures = [se for se in events if se.event.meta.get("ok") is False]
    if not failures:
        lines.append("no failures recorded")
        return "\n".join(lines).rstrip() + "\n"

    lines += [f"{len(failures)} failure(s)", ""]
    for stamped in failures:
        ev = stamped.event
        tool = ev.meta.get("tool", ev.kind)
        name = (ev.meta.get("args") or {}).get("name")
        ident = f"{tool}({name})" if name else str(tool)
        lines += [f"## {stamped.t:%Y-%m-%d %H:%M:%S} {ident}", "", "```", ev.text, "```", ""]

    return "\n".join(lines).rstrip() + "\n"


def render_summary_markdown(
    events: Sequence[StampedEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    funnel_instrumented: bool = True,
    notes: Sequence[str] | None = None,
    segment_label: str | None = None,
) -> str:
    """The cumulative whole-run rollup — same shape as counts, over every event in the run.

    ``notes`` is an optional list of run-level lines rendered under a Notes heading (story #44
    will feed it a self-disabled note); absent notes render no section.
    """
    return _counts_body(
        events,
        window_start=window_start,
        window_end=window_end,
        funnel_instrumented=funnel_instrumented,
        title="QA summary",
        segment_label=segment_label,
        notes=notes,
    )
