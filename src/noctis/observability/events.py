"""Typed observability events + a plain renderer — the neutral seam between a session and its
output.

An :class:`Event` is what the research loop (and, from P3, the trading runtime) hands to its
``on_event`` sink each time something worth watching happens: the model's reasoning, its
narration, a tool call and its gate-facing result, per-round token usage. A
:class:`~noctis.observability.console.Console` colorizes and level-gates events for a terminal;
the logging fallback renders them flat through :func:`render_plain`. This module is **core** —
no heavy deps, no provider imports — so the same event contract holds on every backend.

``level`` is the minimum ``-v`` count at which an event shows (1 = ``-v``, 2 = ``-vv``); the
sink decides. Zero new model calls, zero extra tokens ride on any of this — an event only ever
surfaces what a completion *already returned*.

What a *sink* is lives here too (#334): :class:`EventSink` is the seven-member surface callers
duck-type off ``on_event`` — the call itself, ``delta``/``hint``/``activity``, and the
``verbose``/``show_reasoning``/``saw_think`` reads — and :class:`NullSink` (singleton
:data:`NULL_SINK`) is its safe-default adapter, the shape "no sink" takes. Both sit beside the
:class:`Event` rather than beside a renderer, so a session declares the seam without importing
one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# The kinds the seam knows. think/say/tool/usage are emitted by the research loop (P1);
# author marks one coder completion in the strategy-authoring split (#9); trade/refuse/feed/
# heartbeat are the trading feed (P4); phase frames both. `result` is the outcome half of a
# tool interaction, reserved for a split feed later. `stage` frames the episodic research
# driver's protocol boundaries — FORMULATE/MATCH/AUTHOR/OPTIMIZE/DECIDE (epic #62, #73).
EVENT_KINDS = frozenset(
    {
        "think",
        "say",
        "tool",
        "author",
        "result",
        "usage",
        "trade",
        "refuse",
        "feed",
        "heartbeat",
        "phase",
        "stage",
    }
)

# Plain one-character prefixes per kind (the log / --no-color path). The Console overlays the
# same glyphs with color so colored and plain output read the same.
_PREFIX = {
    "think": "🧠",
    "say": "",
    "tool": "→",
    "author": "✎",
    "result": "←",
    "usage": "·",
    "trade": "$",
    "refuse": "⊘",
    "feed": "~",
    "heartbeat": "♥",
    "phase": "#",
    "stage": "▸",
}


@dataclass(frozen=True)
class Event:
    """One observable moment in a research or trading session.

    ``meta`` carries the structured numbers behind ``text`` (e.g. ``{"gap": 0.1, "ok": True}``)
    so a richer renderer — or a future JSON sink — never has to reparse the prose.
    """

    kind: str
    text: str
    meta: dict = field(default_factory=dict)
    level: int = 1  # minimum -v level at which this event shows


def render_plain(ev: Event) -> str:
    """Render an event to a single uncolored line — the logging fallback and any non-color sink.

    Multi-line ``text`` (reasoning spans paragraphs) is collapsed to one physical line so a log
    keeps one record per event; the Console, which wraps for a terminal, does its own layout.
    """
    prefix = _PREFIX.get(ev.kind, "")
    body = " ".join(ev.text.split())  # collapse newlines/runs of whitespace to a single line
    return f"{prefix} {body}".strip() if prefix else body


# ── the sink contract ────────────────────────────────────────────────────────────────────────
@runtime_checkable
class EventSink(Protocol):
    """What a session's ``on_event`` seam *is* — the seven members its callers duck-type (#334).

    A sink is far more than "a callable": the agent loop reads ``verbose`` to decide whether to
    stream and hands its deltas to ``delta``, brackets each blocking model call in ``activity``,
    and the CLI reads ``saw_think`` afterwards and prints a degradation ``hint``. Declaring all
    seven in one place is what lets a caller *call* instead of probing with ``getattr``, and what
    gives a new adapter — a file sink, a socket sink — something to implement against.

    The call takes an :class:`Event` **or** a bare ``str``: a few legacy sites emit a
    pre-formatted line (a misfire note, a web_search auto-disable), and every adapter renders or
    files it as-is. Implementations conform structurally — nothing subclasses this — so
    :class:`~noctis.observability.console.Console` satisfies it without knowing it exists;
    ``runtime_checkable`` is here so the conformance test can ask an adapter directly.
    """

    verbose: int
    show_reasoning: bool
    saw_think: bool

    def __call__(self, ev: Event | str) -> None:
        """Surface one event (or one legacy pre-formatted line)."""
        ...

    def delta(self, kind: str, text: str) -> None:
        """Surface one streaming ``think``/``say`` delta in place."""
        ...

    def hint(self, text: str) -> None:
        """Surface one advisory line — a graceful-degradation note, not an event."""
        ...

    def activity(self, label: str) -> AbstractContextManager[None]:
        """Bracket a blocking op (a model call, a tool sweep) with a live progress line."""
        ...


class NullSink:
    """The sink that says nothing — the shape "no sink" takes, so no caller has to check for one.

    Every member is a safe default: the call and ``delta``/``hint`` are inert, ``activity`` still
    hands back a ``with``-usable context manager (callers enter it unconditionally), verbosity
    reads as ``0`` — which parks the agent loop's ``verbose >= 2`` streaming gate — and the
    reasoning/saw-think flags read ``False``. Any *other* attribute a future caller reaches for
    resolves to a no-op callable: the point is that nothing a quiet run touches ever raises.

    Stateless by design, so one module-level :data:`NULL_SINK` serves every session, and a
    subclass (the ``--debug`` recorder) inherits the whole console-facing surface for free.
    """

    verbose = 0
    show_reasoning = False
    saw_think = False

    def __call__(self, ev: Event | str) -> None:
        """No-op: with no sink attached there is nothing to surface the event to."""

    def delta(self, kind: str, text: str) -> None:
        """No-op: with no sink attached there is nothing to stream in place."""

    def hint(self, text: str) -> None:
        """No-op: with no sink attached there is no advisory line to print."""

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """A no-op context manager, so ``with sink.activity(...):`` still brackets a blocking
        call for a caller that never asked whether a console was there."""
        yield

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Any other duck-typed attribute resolves to a harmless no-op callable."""
        return lambda *args, **kwargs: None


# One shared, stateless null adapter — it holds no per-session state, so a module singleton is
# enough, and identity (`sink is NULL_SINK`) reads as "this session is quiet".
NULL_SINK = NullSink()


# ── shared event builders ────────────────────────────────────────────────────────────────────
# Both research loops surface the SAME lines through these builders, so a conversation transcript
# and an episodic session read identically. They live here (the neutral seam, no LLM imports) so
# the episodic driver can build tool/stage lines without importing the conversation loop.
def usage_line(usage: dict | None) -> str:
    """A compact per-completion token line from a turn's neutral usage dict (0 for any field a
    provider omits, so a no-usage backend still yields a clean line)."""
    u = usage or {}

    def g(k: str) -> int:
        return int(u.get(k, 0) or 0)

    return (
        f"tokens in={g('input_tokens')} out={g('output_tokens')} "
        f"cache_w={g('cache_creation_input_tokens')} cache_r={g('cache_read_input_tokens')}"
    )


def tool_event(name: str, args: dict, result: dict, brief: dict) -> Event:
    """One :class:`Event` per tool call for logs / the CLI feed (level 1). ``text`` is the call
    plus a compact outcome; ``brief`` — the toolbox's own gate-facing slice of the result — carries
    the numbers, both printed and structured into ``meta``, plus an ``ok`` flag the renderer colors
    on (green success / red error).

    ``meta`` also carries the structured call itself — ``meta["tool"]`` (the name) and
    ``meta["args"]`` (the *same* per-arg-truncated ``brief_args`` shown in ``text``) — so a
    downstream consumer reads the call off the metadata instead of reparsing the prose. Truncation
    happens *before* the args enter ``meta``, so a strategy-source-sized string surfaces only as a
    ``<N chars>`` placeholder and never leaks whole. ``tool``/``args`` are set last, after any
    ``meta.update(brief)``, so the structured keys win deterministically over a same-named brief
    key. The conversation loop and the episodic driver both emit through this one builder, so a
    tool line reads the same whichever loop produced it."""
    brief_args = {
        k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 80 else v)
        for k, v in (args or {}).items()
    }
    meta: dict = {}
    if isinstance(result, dict) and "error" in result:
        meta["ok"] = False
        outcome = f"ERROR: {result['error']}"
    else:
        meta["ok"] = True
        meta.update(brief)
        outcome = ", ".join(f"{k}={v}" for k, v in brief.items()) or "ok"
    meta["tool"] = name
    meta["args"] = brief_args
    text = f"{name}({json.dumps(brief_args, default=str)}) -> {outcome}"
    return Event("tool", text, meta=meta, level=1)


def stage_event(
    stage: str, strategy: str | None = None, *, oracle: list[str] | None = None
) -> Event:
    """One :class:`Event` marking an episodic-driver protocol boundary (#73). The stage label is
    uppercased for a clear boundary line — ``FORMULATE`` / ``MATCH · <strategy>`` — with the
    structured ``stage`` (and ``strategy`` when named) in ``meta``. Level 1, so it shows at ``-v``
    like a tool line: stage boundaries are the episodic session's skeleton, not the ``-vv``
    firehose.

    ``oracle`` (the AUTHOR stage, #86) names the fixed spec's scenarios the candidate is gated
    against — the canonical scenario names from the FORMULATE spec, not a second rendering. When
    present it is both appended to the boundary line (``AUTHOR · <strategy> · oracle: a, b``) and
    carried structurally in ``meta['oracle']``, so an operator audits which oracle a candidate met
    from the live narration. Absent (every other stage) it is omitted, so the line is unchanged."""
    label = stage.upper()
    text = f"{label} · {strategy}" if strategy else label
    meta: dict = {"stage": stage}
    if strategy:
        meta["strategy"] = strategy
    if oracle:
        meta["oracle"] = list(oracle)
        text += f" · oracle: {', '.join(oracle)}"
    return Event("stage", text, meta=meta, level=1)
