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
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The kinds the seam knows. think/say/tool/usage are emitted by the research loop (P1);
# author marks one coder completion in the strategy-authoring split (#9); trade/refuse/feed/
# heartbeat are the trading feed (P4); phase frames both. `result` is the outcome half of a
# tool interaction, reserved for a split feed later.
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
