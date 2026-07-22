"""A level-aware, colorized, width-capped console sink for observability events.

The CLI builds one and passes it as the research loop's ``on_event``. It accepts either an
:class:`~noctis.observability.events.Event` — colorized by ``kind`` and gated by ``-v`` level —
or a bare ``str`` (a legacy pre-formatted line, shown as-is), so old emit sites keep working
untouched. Coloring uses ``typer`` (already a core dep); ``color=False`` (or the ``NO_COLOR``
env var, or a non-tty stream via ``typer.echo``) drops every escape.

Level model: an event shows when ``verbose >= event.level``. ``--show-reasoning`` is the one
exception — it forces ``think``/``say`` regardless of ``-v`` so an operator can watch the model
reason without opening the full ``-vv`` firehose.

Token streaming (P5): when attached to a TTY, :meth:`Console.delta` renders ``think``/``say``
deltas in place as they arrive, and the matching completed block :class:`Event` is then dropped
(the live stream already showed that text). Off a TTY — a pipe, a log, a redirect — ``delta`` is
a clean no-op and the completed block renders normally, so nothing depends on streaming.
"""

from __future__ import annotations

import os
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import typer

from noctis.observability.events import Event

# Per-kind terminal style: (glyph, typer color | None, dim?). The glyph mirrors events._PREFIX
# so colored and plain output line up. `tool`/`result` recolor to red on an error result.
_STYLE: dict[str, tuple[str, str | None, bool]] = {
    "think": ("🧠", typer.colors.CYAN, True),
    "say": ("", None, False),
    "tool": ("→", typer.colors.GREEN, False),
    "author": ("✎", typer.colors.CYAN, False),
    "result": ("←", typer.colors.GREEN, False),
    "usage": ("·", None, True),
    "trade": ("$", typer.colors.GREEN, False),
    "refuse": ("⊘", typer.colors.YELLOW, False),
    "feed": ("~", None, True),
    "heartbeat": ("♥", typer.colors.BLUE, True),
    "phase": ("#", typer.colors.MAGENTA, False),
    # A stage boundary is the episodic session's inner frame (#73): the `▸` glyph + a distinct
    # bright-magenta so it stands apart from the outer `phase` frame while reading as its sibling.
    "stage": ("▸", typer.colors.BRIGHT_MAGENTA, False),
}

_MIN_BODY_WIDTH = 20  # never wrap narrower than this, however deep the gutter

# Heartbeat (P6): a live in-place activity line while a blocking model call or tool sweep runs,
# so -v isn't silent for the minutes one can take. Braille spinner frames + a repaint interval.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ACTIVITY_INTERVAL = 0.125  # seconds between spinner repaints


def _stdout_write(text: str) -> None:
    """The default newline-free writer for token streaming: write and flush stdout so a delta
    shows the instant it arrives (a typewriter feel), unlike ``typer.echo`` which appends a line."""
    sys.stdout.write(text)
    sys.stdout.flush()


class Console:
    """Level-aware, colorized, width-capped event sink; falls back to plain ``typer.echo``."""

    def __init__(
        self,
        verbose: int,
        *,
        show_reasoning: bool = False,
        width: int = 100,
        color: bool = True,
        sink: Callable[[str], None] | None = None,
        stream_sink: Callable[[str], None] | None = None,
        tty: bool | None = None,
    ) -> None:
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.width = width
        self.color = color and not os.getenv("NO_COLOR")
        self._sink = sink or typer.echo
        # A newline-free raw writer for in-place token streaming (P5); the line-appending _sink
        # can't stream. Defaults to stdout so the typewriter feel works in a real terminal.
        self._stream_sink = stream_sink or _stdout_write
        # Streaming only makes sense on an interactive terminal; a pipe/redirect/log falls back
        # to block rendering. Detected from stdout unless the caller pins it (tests do).
        if tty is None:
            isatty = getattr(sys.stdout, "isatty", None)
            self._tty = bool(isatty and isatty())
        else:
            self._tty = tty
        # Whether any `think` event was actually surfaced this session — drives the CLI's
        # "reasoning not surfaced by <provider>" hint when a reasoning view came back empty.
        self.saw_think = False
        # In-place stream state (P5): the kind of the open stream (or None), the current column
        # for width-wrapping, the continuation indent, and the paint fn for the open kind.
        self._stream_kind: str | None = None
        self._stream_col = 0
        self._stream_indent = ""
        self._stream_paint: Callable[[str], str] = lambda s: s
        # Kinds streamed live since their completed block last arrived — the block for a kind in
        # here is the duplicate of a live stream and is dropped (a block consumes its own flag).
        self._streamed: set[str] = set()
        # Serializes the heartbeat thread's in-place repaints against its own final erase (P6).
        self._activity_lock = threading.Lock()

    def _visible(self, ev: Event) -> bool:
        # --show-reasoning forces think/say; every other kind (and think/say without it) follows -v.
        if self.show_reasoning and ev.kind in ("think", "say"):
            return True
        return self.verbose >= ev.level

    def __call__(self, ev: Event | str) -> None:
        if isinstance(ev, str):
            # A legacy pre-formatted line (a misfire note, web_search auto-disable): shown as-is.
            self._finalize_stream()  # close any open live stream before the line prints
            self._sink(ev)
            return
        if ev.kind in self._streamed:
            # This kind already rendered live this round as a token stream, so the completed
            # block is a duplicate — finalize the stream and drop the block. Independent of the
            # -v gate: streaming only runs where the block would show, so the flag is authoritative.
            self._streamed.discard(ev.kind)
            self._finalize_stream()
            if ev.kind == "think":
                self.saw_think = True
            return
        if not self._visible(ev):
            return
        self._finalize_stream()  # a stream of another kind ends before an unrelated block prints
        if ev.kind == "think":
            self.saw_think = True
        self._sink(self._format(ev))

    def delta(self, kind: str, text: str) -> None:
        """Render one streaming ``think``/``say`` delta in place (P5). A no-op unless attached to
        a TTY — a piped/redirected stream falls back to block mode, where the loop's completed
        Event renders the same text a moment later — and for any kind other than think/say."""
        if not self._tty or kind not in ("think", "say") or not text:
            return
        if self._stream_kind is not None and self._stream_kind != kind:
            self._finalize_stream()  # a kind switch (think → say) closes the previous line first
        if self._stream_kind is None:
            self._begin_stream(kind)
        self._streamed.add(kind)
        self._stream_append(text)

    def hint(self, text: str) -> None:
        """Print one dim advisory line (graceful-degradation hint), color permitting."""
        self._finalize_stream()
        line = f"({text})"
        self._sink(typer.style(line, dim=True) if self.color else line)

    # ── in-place activity heartbeat (P6) ─────────────────────────────────────────────────────────
    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """A live in-place progress line for a blocking op — a model call or a tool sweep — so
        ``-v`` isn't silent for the minutes one can take. On a TTY: an animated spinner + elapsed
        clock that erases itself when the op returns (the real result Event prints next). Off a
        TTY — a pipe, a redirect, a log — one dim ``<label> …`` breadcrumb at the start and nothing
        after, so logs record what ran without spinner-frame spam. Bracket the blocking call:
        ``with console.activity(label): result = do_work()``."""
        self._finalize_stream()  # close any open token stream before the activity line opens
        label = " ".join(label.split())  # collapse whitespace so the line stays a single row
        if not self._tty:
            line = f"⠿ {label} …"
            self._sink(typer.style(line, fg=typer.colors.BLUE, dim=True) if self.color else line)
            yield
            return
        stop = threading.Event()
        started = time.monotonic()

        def _animate() -> None:
            i = 0
            while not stop.wait(_ACTIVITY_INTERVAL):
                frame = self._activity_frame(label, int(time.monotonic() - started), i)
                i += 1
                with self._activity_lock:
                    self._stream_sink("\r" + frame + "\x1b[K")

        thread = threading.Thread(target=_animate, name="noctis-activity", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            with self._activity_lock:
                self._stream_sink("\r\x1b[K")  # erase the spinner; the result Event prints next

    def _activity_frame(self, label: str, elapsed_s: int, i: int) -> str:
        """One spinner line — ``<frame> <label> · m:ss``, truncated to ``width``. Pure: the
        animation thread and the tests both build frames here, so the format is asserted without
        threads or timing."""
        frame = _SPINNER[i % len(_SPINNER)]
        text = f"{frame} {label} · {elapsed_s // 60}:{elapsed_s % 60:02d}"
        if len(text) > self.width:
            text = text[: self.width - 1] + "…"
        return typer.style(text, fg=typer.colors.BLUE, dim=True) if self.color else text

    # ── in-place token streaming (P5) ──────────────────────────────────────────────────────────
    def _begin_stream(self, kind: str) -> None:
        """Open an in-place stream for ``kind``: write its glyph gutter once and pin the paint fn
        and continuation indent, mirroring the block renderer so live and block output line up."""
        glyph, color, dim = _STYLE.get(kind, ("", None, False))
        gutter = f"{glyph} " if glyph else ""
        self._stream_kind = kind
        self._stream_indent = " " * len(gutter)
        self._stream_col = len(gutter)
        if self.color:
            self._stream_paint = lambda s: typer.style(s, fg=color, dim=dim)
        else:
            self._stream_paint = lambda s: s
        if gutter:
            self._stream_sink(self._stream_paint(gutter))

    def _stream_append(self, text: str) -> None:
        """Emit a delta immediately (a real typewriter feel — nothing is held back), soft-wrapping
        at spaces once the line passes ``width`` and honoring source newlines with an aligned
        indent. Runs between break points are written as one painted chunk to keep escapes sane."""
        run: list[str] = []

        def flush_run() -> None:
            if run:
                self._stream_sink(self._stream_paint("".join(run)))
                run.clear()

        for ch in text:
            if ch == "\n":
                flush_run()
                self._stream_sink(self._stream_paint("\n" + self._stream_indent))
                self._stream_col = len(self._stream_indent)
            elif ch == " " and self._stream_col >= self.width:
                flush_run()  # wrap at the space instead of printing it, once past the width cap
                self._stream_sink(self._stream_paint("\n" + self._stream_indent))
                self._stream_col = len(self._stream_indent)
            else:
                run.append(ch)
                self._stream_col += 1
        flush_run()

    def _finalize_stream(self) -> None:
        """Close the open stream (if any) with a trailing newline so the next output starts on a
        fresh line. Idempotent — a no-op when nothing is streaming."""
        if self._stream_kind is None:
            return
        self._stream_sink("\n")
        self._stream_kind = None
        self._stream_col = 0

    def _format(self, ev: Event) -> str:
        glyph, color, dim = _STYLE.get(ev.kind, ("", None, False))
        if ev.kind in ("tool", "result", "author") and ev.meta.get("ok") is False:
            color = typer.colors.RED
        gutter = f"{glyph} " if glyph else ""
        wrapped = self._wrap(ev.text, gutter)
        if not self.color:
            return wrapped
        return typer.style(wrapped, fg=color, dim=dim)

    def _wrap(self, text: str, gutter: str) -> str:
        """Wrap ``text`` to ``width``, giving the first physical line the ``gutter`` glyph and
        every continuation an equal-width indent so multi-line reasoning stays aligned."""
        indent = " " * len(gutter)
        body_width = max(self.width - len(gutter), _MIN_BODY_WIDTH)
        out: list[str] = []
        for src in text.rstrip().splitlines() or [""]:
            for seg in textwrap.wrap(src, width=body_width) or [""]:
                out.append((gutter if not out else indent) + seg)
        return "\n".join(out)
