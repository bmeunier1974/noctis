"""The fail-safe latch — one trip/warn-once/latch-off contract, composed rather than re-typed.

Noctis writes a lot of *secondary* evidence: the debug recorder's hour-segmented report tree, the
capture store's content-addressed sidecars. None of it is the trading system; all of it is written
from inside the engine's own call stack. So every one of those writers owes the engine the same
promise, and it is a promise with three clauses:

* **Trip on the first internal failure.** Not the second, not after a retry — a writer that
  cannot write once is a writer with a broken disk, a full volume or a revoked permission, and
  trying again just multiplies the damage while the engine waits.
* **Warn exactly once, naming what stopped.** One line an operator can act on: which writer went
  quiet, what failed, and what that costs them for the rest of the session. Never a warning per
  failed write — a wedged disk would drown the log the run's real signal lives in.
* **Latch off to silent no-ops, and never raise.** Once off it stays off, so a session reads as
  one honest behaviour rather than intermittent half-coverage, and *nothing* propagates into the
  engine (AGENTS.md rule 2, in spirit: a debug tool must never degrade or crash a run).

This module owns that contract once. :class:`FailSafe` is composed — by
:class:`~noctis.observability.debug.recorder.Recorder`, by
:class:`~noctis.observability.capture.CaptureStore` — never re-implemented, so "warn once then go
quiet" cannot drift into three subtly different behaviours in three files.

**The writer is a seam** (``Callable[[Path, str], None]``, default :func:`write_text`). A composer
routes its disk touches through :meth:`FailSafe.write` and inherits the encoding decision made
here, once; a test hands the latch a writer that raises and exercises the fail-safe path through
the composer's own public API — no process-wide patching of ``Path.write_text``, which is a blunt
instrument that catches pytest's own I/O as readily as the code under test.

**Two deliberate asymmetries**, both load-bearing:

* :meth:`write` is *not* gated on the latch — gating is :meth:`guard`'s job. That is what lets a
  composer stamp a best-effort honesty note (``on_trip``) *after* the latch has already tripped,
  and it is what makes a failed write inside a guarded body abort the rest of that body instead of
  letting it carry on to write a stale artifact that reads as a whole one.
* Only ``Exception`` is caught. ``KeyboardInterrupt`` is the operator talking to the engine, not
  an internal failure: it rides straight through, and it never latches observability off.

Reading nothing is gated either — a latch stops *writing*. What a writer put on disk before it
failed is exactly what a post-mortem wants back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

__all__ = ["FailSafe", "Writer", "write_text"]

T = TypeVar("T")

# The seam a composer's disk touches ride: given a path and a payload, put the payload there.
Writer = Callable[[Path, str], None]


def write_text(path: Path, payload: str) -> None:
    """The default writer: the payload's UTF-8 bytes at ``path``, the one encoding decision."""
    path.write_text(payload, encoding="utf-8")


class FailSafe:
    """A one-way latch around secondary work: transparent until it fails, silent forever after.

    Construct one per writer, naming the three things the single warning is made of:

    * ``subject`` — *what* stopped, in the operator's terms (``"debug recorder <run-id>"``,
      ``"capture store <root>"``). It is the writer's identity, not the module's.
    * ``cause`` — the kind of failure, as a noun phrase (default ``"an internal failure"``).
    * ``consequence`` — what the operator loses for the rest of the session.

    …plus the ``logger`` to say it through (a composer passes **its own** module logger, so the
    warning surfaces under the name an operator would grep for), the ``writer`` seam, and an
    optional ``on_trip`` note — a best-effort honesty stamp run once, after the warning, whose own
    failure is swallowed silently because the latch is already set and a second warning would be
    noise on top of a writer that is already off.

    Surface: :meth:`guard` (run something behind the latch), :meth:`write` (write through the
    seam), :attr:`tripped` (has it fired — read-only, never clears).
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        subject: str,
        consequence: str,
        cause: str = "an internal failure",
        writer: Writer = write_text,
        on_trip: Callable[[], None] | None = None,
    ) -> None:
        self._logger = logger
        self._subject = subject
        self._consequence = consequence
        self._cause = cause
        self._writer = writer
        self._on_trip = on_trip
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """Whether the latch has fired. Once ``True`` it never clears."""
        return self._tripped

    def guard(self, work: Callable[..., T], *args: object) -> T | None:
        """Run ``work(*args)`` behind the latch and hand back what it returned.

        A no-op returning ``None`` once the latch has tripped; otherwise the first ``Exception``
        out of ``work`` is funnelled into the trip (which warns, latches and swallows) and the
        caller is told ``None`` — the honest signal that nothing was done, so it can omit a field
        rather than name an artifact that was never written.

        WHY one choke point: a composer's future public method cannot forget the guard if the only
        way its body runs is through here.
        """
        if self._tripped:
            return None
        try:
            return work(*args)
        except Exception as exc:
            self._trip(exc)
            return None

    def write(self, path: Path, payload: str) -> None:
        """Write ``payload`` to ``path`` through the injected writer.

        Deliberately **not** gated on the latch, and deliberately allowed to raise: inside a
        guarded body the failure reaches that body's own :meth:`guard`, which trips exactly once
        and abandons the rest of the body; outside one — the ``on_trip`` note — it is the only way
        a tripped writer can still stamp the truth about its own silence.
        """
        self._writer(path, payload)

    def _trip(self, exc: BaseException) -> None:
        """Fire the latch: one warning naming what stopped, then silence, then the note."""
        if self._tripped:
            return
        self._tripped = True
        self._logger.warning(
            "%s self-disabled after %s (%s: %s); %s",
            self._subject,
            self._cause,
            type(exc).__name__,
            exc,
            self._consequence,
        )
        if self._on_trip is not None:
            try:
                self._on_trip()
            except Exception:
                pass  # best effort only: the latch is set; a failed note earns no second warning
