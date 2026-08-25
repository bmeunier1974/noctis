"""Observability seam: typed :class:`Event`s + a level-aware :class:`Console` renderer.

One import surface for a session's output layer. Loops emit ``Event``s to an ``on_event`` sink;
the CLI builds a ``Console``, the runtime and tests can pass a plain callable, and the logging
fallback renders via :func:`render_plain`. When a recorder needs to ride alongside the console on
that single sink, :class:`EventTee` splits each event to both while delegating its primary's
duck-typed surface. Core only — no provider SDKs.

What every one of those adapters *is* — the seven members a caller reads off the seam — is
declared once as the :class:`EventSink` Protocol, with :class:`NullSink` (singleton
:data:`NULL_SINK`) as the quiet adapter a session holds when nothing is watching.
"""

from __future__ import annotations

from noctis.observability.console import Console
from noctis.observability.events import (
    EVENT_KINDS,
    NULL_SINK,
    Event,
    EventSink,
    NullSink,
    render_plain,
)
from noctis.observability.tee import EventTee

__all__ = [
    "Console",
    "EVENT_KINDS",
    "Event",
    "EventSink",
    "EventTee",
    "NULL_SINK",
    "NullSink",
    "render_plain",
]
