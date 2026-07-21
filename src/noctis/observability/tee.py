"""An :class:`EventTee` — one recorder rides alongside the console on the single ``on_event`` seam.

The ``--debug`` QA run (epic #36) wants every observability event in two places at once: the
level-aware :class:`~noctis.observability.console.Console` a watcher reads, *and* a recorder that
files the run's hour-segmented report. The seam is a single ``on_event`` callable, and callers
duck-type far more than "a callable" off it — the agent loop reads ``verbose``/``activity``/
``delta`` to decide whether to stream, the CLI reads ``saw_think`` and calls ``hint``. So a naive
``lambda ev: (console(ev), recorder(ev))`` breaks every one of those call sites the moment it
replaces the console.

:class:`EventTee` is the honest splitter: calling it forwards the event to the primary console
first (unguarded — a console bug should surface loudly), then to each secondary inside a guard so
a raising recorder can never break the console path or its siblings. *Every other* attribute
access delegates to the primary console, so the whole duck-typed surface keeps working. When the
run is quiet (no ``-v``) but a recorder is attached, there is no console to delegate to — the
primary is ``None`` and delegation resolves against :data:`_NULL_CONSOLE`, whose every attribute
is a safe no-op, so a bare ``--debug`` run records without a single access raising.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from noctis.observability.console import Console
from noctis.observability.events import Event

logger = logging.getLogger("noctis.observability.tee")

# A secondary sink is any event callable — a recorder lands in a later story, so this is typed
# generically. It takes the same ``Event | str`` the console sink takes.
EventSink = Callable[["Event | str"], None]


class _NullConsole:
    """A safe no-op stand-in for an absent primary console.

    When a recorder rides the seam on a quiet run — ``--debug`` with no ``-v`` — there is no
    :class:`Console` to delegate to, yet callers still duck-type the console surface off the tee.
    Every attribute they touch resolves here to a harmless default so no access ever raises:
    verbosity reads as ``0`` (which parks the agent loop's ``verbose >= 2`` streaming gate),
    the reasoning/saw-think flags read ``False``, ``delta``/``hint`` are inert calls, and
    ``activity`` still hands back a ``with``-usable context manager. Any *other* attribute a
    future caller reaches for resolves to a no-op callable — the point is that nothing raises.
    """

    verbose = 0
    show_reasoning = False
    saw_think = False

    def delta(self, kind: str, text: str) -> None:
        """No-op: with no console there is nothing to stream in place."""

    def hint(self, text: str) -> None:
        """No-op: with no console there is no advisory line to print."""

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """A no-op context manager so ``with sink.activity(...):`` still brackets a blocking call
        (the agent loop enters the result of ``on_event.activity(label)`` unconditionally)."""
        yield

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Any other duck-typed attribute resolves to a harmless no-op callable."""
        return lambda *args, **kwargs: None


# One shared, stateless stand-in — it holds no per-session state, so a module singleton is enough.
_NULL_CONSOLE = _NullConsole()


class EventTee:
    """Forward each event to a primary console then to each guarded secondary; delegate the rest.

    ``primary`` is the level-aware console (or ``None`` on a quiet run that only records);
    ``secondaries`` are recorder-style event sinks. Calling the tee renders on the primary, then
    hands the same event to every secondary — each inside a guard, so a raising recorder is logged
    once and skipped, never breaking the console path or a later secondary. Any attribute other
    than the call itself delegates to the primary (its ``delta``/``activity``/``hint`` methods and
    its ``verbose``/``show_reasoning``/``saw_think`` reads), falling back to :data:`_NULL_CONSOLE`
    when there is no primary so the whole surface stays safe.
    """

    def __init__(self, primary: Console | None, *secondaries: EventSink) -> None:
        self._primary = primary
        self._secondaries = secondaries

    def __call__(self, ev: Event | str) -> None:
        if self._primary is not None:
            self._primary(ev)  # unguarded: a console bug should surface, not be swallowed
        for secondary in self._secondaries:
            try:
                secondary(ev)
            except Exception:
                # A recorder failure is isolated: log it once and move on so the console path and
                # every other secondary still see the event.
                logger.warning(
                    "observability secondary sink raised; primary path unaffected", exc_info=True
                )

    def __getattr__(self, name: str) -> object:
        # Reached only for attributes the tee does not define itself — i.e. the console surface a
        # caller duck-types (delta, hint, activity, verbose, show_reasoning, saw_think, …).
        # Delegate to the primary, or to the safe no-op stand-in when there is no console.
        target = self.__dict__.get("_primary") or _NULL_CONSOLE
        return getattr(target, name)
