"""An :class:`EventTee` — one recorder rides alongside the console on the single ``on_event`` seam.

The ``--debug`` QA run (epic #36) wants every observability event in two places at once: the
level-aware :class:`~noctis.observability.console.Console` a watcher reads, *and* a recorder that
files the run's hour-segmented report. The seam is a single ``on_event`` sink, and callers
duck-type far more than "a callable" off it — the agent loop reads ``verbose``/``activity``/
``delta`` to decide whether to stream, the CLI reads ``saw_think`` and calls ``hint``. So a naive
``lambda ev: (console(ev), recorder(ev))`` breaks every one of those call sites the moment it
replaces the console.

:class:`EventTee` is the honest splitter: calling it forwards the event to the primary sink first
(unguarded — a console bug should surface loudly), then to each secondary inside a guard so a
raising recorder can never break the primary path or its siblings. *Every other* attribute access
delegates to the primary, so the whole duck-typed surface keeps working.

The primary is always a real :class:`~noctis.observability.events.EventSink` (#335): a quiet run
(no ``-v``, recorder attached) hands the tee
:data:`~noctis.observability.events.NULL_SINK` rather than nothing at all, so the tee never asks
whether it has a primary — the call is inert and every attribute is a safe no-op, and a bare
``--debug`` run records without a single access raising. One null adapter serves the whole
package; the tee keeps no private stand-in of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from noctis.observability.events import Event, EventSink

logger = logging.getLogger("noctis.observability.tee")


class EventTee:
    """Forward each event to a primary sink then to each guarded secondary; delegate the rest.

    ``primary`` is the session's real sink — the level-aware console, or
    :data:`~noctis.observability.events.NULL_SINK` on a quiet run that only records.
    ``secondaries`` stay bare event callables: a secondary is only ever *called*, so anything that
    takes an ``Event | str`` (a recorder, a list's ``append``) rides along without conforming to
    the seven-member seam. Calling the tee renders on the primary, then hands the same event to
    every secondary — each inside a guard, so a raising recorder is logged once and skipped, never
    breaking the primary path or a later secondary. Any attribute other than the call itself
    delegates to the primary (its ``delta``/``activity``/``hint`` methods and its
    ``verbose``/``show_reasoning``/``saw_think`` reads), which makes the tee itself an
    :class:`~noctis.observability.events.EventSink` whatever it is teeing.
    """

    def __init__(self, primary: EventSink, *secondaries: Callable[[Event | str], None]) -> None:
        self._primary = primary
        self._secondaries = secondaries

    def __call__(self, ev: Event | str) -> None:
        self._primary(ev)  # unguarded: a console bug should surface, not be swallowed
        for secondary in self._secondaries:
            try:
                secondary(ev)
            except Exception:
                # A recorder failure is isolated: log it once and move on so the primary path and
                # every other secondary still see the event.
                logger.warning(
                    "observability secondary sink raised; primary path unaffected", exc_info=True
                )

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes the tee does not define itself — i.e. the sink surface a
        # caller duck-types (delta, hint, activity, verbose, show_reasoning, saw_think, …).
        # Delegation is unconditional: the primary is a real sink, so whatever it answers is the
        # answer — including a primary that reads as falsy. (The read goes through ``__dict__``
        # so a half-built tee cannot recurse back into this method.) The return is ``Any`` — the
        # honest type for a delegating proxy — so a caller can invoke the proxied methods
        # (``sink.hint(...)``) without the type checker rejecting the call.
        return getattr(self.__dict__["_primary"], name)
