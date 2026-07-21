"""The recorder — the one disk-touching module in the debug subpackage (story #43, epic #36).

Everything else under ``noctis.observability.debug`` is pure: the funnel ledger and the four
renderers are functions over a hand-built event list, no clock and no I/O. The recorder is the
seam that gives them a clock and a filesystem. It owns the per-run report tree
(``<qa_dir>/<run-id>/``), the elapsed-hour segment lifecycle, arrival-timestamping of events into
``events.jsonl``, the cumulative ``summary.md``, the ``run.json`` manifest, and ``close()`` — and
it renders *through* the pure layer, holding no funnel or markdown logic of its own.

**Why there is no background thread — this is load-bearing, not a style choice.** The four PRs
just before this one (#30–#35) existed to *remove* shutdown join hazards: the bounded pool
teardown that never joins a wedged worker. A debug writer that spun up a background flush thread
would reintroduce exactly the failure those PRs closed — a stop that blocks forever on a writer
that will not join. So every write here is **synchronous** on the calling thread: appends are
buffered in memory and flushed to disk at three explicit points — an hour rollover, a phase
transition, and ``close()`` — with no thread, no ``asyncio``, and no per-event ``fsync`` anywhere
in this module. If the process is killed mid-run, the worst loss is the current hour's unflushed
tail; the manifest and every finalized segment are already on disk.

**The clock is injected** (``Callable[[], datetime]``) so tests are deterministic and no
``datetime.now()`` is ever reached: construction reads it once for ``started``, each event reads
it once for its arrival stamp, ``close()`` reads it once for ``stopped``. The manifest fields the
recorder cannot know (argv, mode, config digest, versions) are injected too — the recorder owns
only ``run_id`` and the ``started``/``stopped``/``duration_s`` stamps, and never computes a digest.

**Segments are ELAPSED hours since start.** On each event ``h = int(elapsed // 3600)``; the first
event of an hour lazily creates ``h{h:02d}/`` (an idle hour writes nothing, so ``h`` may jump
0 → 3 and only ``h00``/``h03`` exist). A rollover finalizes the previous segment — its
``counts.md``/``counts.json``/``errors.md``/``events.jsonl`` rendered from *that hour's* events
over that hour's UTC window — then rewrites ``summary.md`` from *all* events so far. Per-hour
counters therefore reset each segment while the summary holds running totals.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from noctis.observability.debug.funnel import StampedEvent
from noctis.observability.debug.render import (
    render_counts_json,
    render_counts_markdown,
    render_errors_markdown,
    render_summary_markdown,
)
from noctis.observability.events import Event

__all__ = ["Recorder"]

# Seconds in an elapsed-hour segment. Named so the rollover arithmetic reads as intent.
_SEGMENT_SECONDS = 3600


def _utc_millis_iso(dt: datetime) -> str:
    """``2026-07-20T14:51:02.418Z`` — UTC ISO-8601 with millisecond precision and a ``Z`` marker.

    The frozen :class:`Event` carries no timestamp, so this is the recorder's arrival stamp shape
    for both the ``events.jsonl`` ``t`` field and the manifest ``started``/``stopped`` stamps. An
    aware datetime is normalized to UTC so the trailing ``Z`` stays honest; a naive one is taken
    as UTC as-is (the whole subpackage treats its stamps as UTC wall-clock by contract).
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03d}Z"


def _as_event(ev: Event | str) -> Event:
    """Accept both shapes the primary sink accepts.

    Per :meth:`Console.__call__`, a secondary receives whatever the primary receives — an
    :class:`Event`, or a legacy pre-formatted ``str`` line (a misfire note, a web_search
    auto-disable). A bare string is wrapped honestly as a ``feed`` event with its text and empty
    ``meta``, so it lands in the jsonl as a real record rather than being silently dropped.
    """
    if isinstance(ev, Event):
        return ev
    return Event("feed", str(ev), meta={})


class Recorder:
    """Files a debug run's hour-segmented report tree under ``<qa_dir>/<run-id>/``.

    Public surface (kept minimal for stories #44/#45 to build on):

    * ``__call__(ev)`` — record one event (an :class:`Event` or a legacy ``str`` line). This is
      the callable the tee (#40) hands each observability event to.
    * ``flush()`` — write the current segment's buffered ``events.jsonl`` to disk. Called
      internally at every phase transition; exposed so #45 can force a flush if it needs one.
    * ``mark_legacy_research()`` — switch subsequent renders to ``funnel_instrumented=False`` (the
      honesty line instead of a zero-filled funnel), for a legacy proposer/Optuna session.
    * ``close()`` — finalize the open segment, rewrite ``summary.md``, stamp ``run.json`` with
      ``stopped``/``duration_s``. Idempotent: a second call is a harmless no-op.

    Construction stamps ``started`` from the clock and writes ``run.json`` immediately (with
    ``stopped``/``duration_s`` null), so a crashed run still leaves an honest manifest.
    """

    def __init__(
        self,
        qa_dir: Path | str,
        *,
        run_id: str,
        clock: Callable[[], datetime],
        manifest: dict | None = None,
    ) -> None:
        self._run_dir = Path(qa_dir) / run_id
        self._run_id = run_id
        self._clock = clock
        self._manifest = dict(manifest or {})

        self._started = clock()
        self._stopped: datetime | None = None
        self._funnel_instrumented = True

        # The current phase, tracked off ``phase`` frames' ``meta["phase"]`` for the jsonl ``phase``
        # field. ``None`` before the first phase frame — an honest placeholder, kept stable.
        self._phase: str | None = None

        # The open segment (``None`` before the first event) and its buffered contents. ``_all``
        # is the whole-run stream the cumulative summary renders from.
        self._current_hour: int | None = None
        self._hour_events: list[StampedEvent] = []
        self._hour_lines: list[str] = []
        self._all: list[StampedEvent] = []

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    # ── the event sink ────────────────────────────────────────────────────────────────────────

    def __call__(self, ev: Event | str) -> None:
        """Record one event: arrival-stamp it, roll the segment if the hour advanced, buffer it.

        Ignores events after :meth:`close` — the run is over and its manifest is final.
        """
        if self._stopped is not None:
            return

        event = _as_event(ev)
        now = self._clock()
        elapsed = (now - self._started).total_seconds()
        h = max(0, int(elapsed // _SEGMENT_SECONDS))

        # Update the tracked phase before stamping so a phase frame's own line carries the phase
        # it announces (entering RESEARCH is stamped RESEARCH, not the prior phase).
        if event.kind == "phase":
            phase = event.meta.get("phase")
            if phase is not None:
                self._phase = str(phase)

        stamped = StampedEvent(now, event)
        line = self._jsonl_line(now, elapsed, event)

        rolled = False
        if self._current_hour is None:
            self._open_segment(h)
        elif h > self._current_hour:
            self._finalize_segment(self._current_hour)
            self._open_segment(h)
            rolled = True

        self._hour_events.append(stamped)
        self._hour_lines.append(line)
        self._all.append(stamped)

        # A phase transition is a flush point: push the buffered jsonl to disk now.
        if event.kind == "phase":
            self.flush()
        # The cumulative summary is rewritten at each rollover (and, below, at close).
        if rolled:
            self._write_summary(now)

    def flush(self) -> None:
        """Write the current segment's buffered ``events.jsonl`` to disk (a no-op if none open)."""
        if self._current_hour is not None:
            self._write_jsonl(self._current_hour, self._hour_lines)

    def mark_legacy_research(self) -> None:
        """Render subsequent counts/summary as legacy (uninstrumented) — the honesty line, no
        zero-filled funnel. Set once when a session runs the legacy proposer/Optuna loop."""
        self._funnel_instrumented = False

    def close(self) -> None:
        """Finalize the open segment, rewrite ``summary.md``, stamp ``run.json``. Idempotent."""
        if self._stopped is not None:
            return
        now = self._clock()
        self._stopped = now
        if self._current_hour is not None:
            self._finalize_segment(self._current_hour)
        self._write_summary(now)
        self._write_manifest()

    # ── segment lifecycle ──────────────────────────────────────────────────────────────────────

    def _open_segment(self, h: int) -> None:
        """Start (lazily create) elapsed-hour segment ``h`` and reset its per-hour buffers."""
        self._current_hour = h
        self._hour_events = []
        self._hour_lines = []
        (self._run_dir / f"h{h:02d}").mkdir(parents=True, exist_ok=True)

    def _finalize_segment(self, h: int) -> None:
        """Write segment ``h``'s four documents from *this hour's* buffered events and window."""
        seg = self._run_dir / f"h{h:02d}"
        window_start = self._started + timedelta(seconds=h * _SEGMENT_SECONDS)
        window_end = self._started + timedelta(seconds=(h + 1) * _SEGMENT_SECONDS)
        label = f"h{h:02d}"
        events = self._hour_events

        self._write_jsonl(h, self._hour_lines)
        (seg / "counts.md").write_text(
            render_counts_markdown(
                events,
                window_start=window_start,
                window_end=window_end,
                funnel_instrumented=self._funnel_instrumented,
                segment_label=label,
            ),
            encoding="utf-8",
        )
        doc = render_counts_json(
            events,
            window_start=window_start,
            window_end=window_end,
            funnel_instrumented=self._funnel_instrumented,
            segment_label=label,
        )
        (seg / "counts.json").write_text(
            json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (seg / "errors.md").write_text(
            render_errors_markdown(
                events,
                window_start=window_start,
                window_end=window_end,
                segment_label=label,
            ),
            encoding="utf-8",
        )

    def _write_jsonl(self, h: int, lines: list[str]) -> None:
        (self._run_dir / f"h{h:02d}" / "events.jsonl").write_text("".join(lines), encoding="utf-8")

    # ── whole-run documents ────────────────────────────────────────────────────────────────────

    def _write_summary(self, now: datetime) -> None:
        """Rewrite ``summary.md`` — the cumulative rollup over every event so far, window
        ``[started, now]``."""
        (self._run_dir / "summary.md").write_text(
            render_summary_markdown(
                self._all,
                window_start=self._started,
                window_end=now,
                funnel_instrumented=self._funnel_instrumented,
            ),
            encoding="utf-8",
        )

    def _write_manifest(self) -> None:
        """Write ``run.json``: the injected fields plus the recorder-owned run id and stamps."""
        data = dict(self._manifest)
        data["run_id"] = self._run_id
        data["started"] = _utc_millis_iso(self._started)
        if self._stopped is not None:
            data["stopped"] = _utc_millis_iso(self._stopped)
            data["duration_s"] = round((self._stopped - self._started).total_seconds(), 3)
        else:
            data["stopped"] = None
            data["duration_s"] = None
        (self._run_dir / "run.json").write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )

    # ── the jsonl line ─────────────────────────────────────────────────────────────────────────

    def _jsonl_line(self, now: datetime, elapsed: float, event: Event) -> str:
        """Build one arrival-stamped ``events.jsonl`` line (a trailing newline included).

        Shape (documented in the epic): ``t`` (UTC ISO-8601 millis + ``Z``), ``el`` (seconds since
        start, one decimal), ``phase`` (tracked current phase or ``null``), ``kind``, then ``tool``
        and ``ok`` lifted from ``meta`` *only when present* (tool events), then ``text`` and
        ``meta`` always. Key order matches the epic's documented line.
        """
        meta = event.meta or {}
        obj: dict = {
            "t": _utc_millis_iso(now),
            "el": round(elapsed, 1),
            "phase": self._phase,
            "kind": event.kind,
        }
        if "tool" in meta:
            obj["tool"] = meta["tool"]
        if "ok" in meta:
            obj["ok"] = meta["ok"]
        obj["text"] = event.text
        obj["meta"] = meta
        return json.dumps(obj, default=str) + "\n"
