"""The observability tee (epic #36): an :class:`EventTee` that rides a recorder alongside the
console on the one ``on_event`` seam, plus the event-sink builder that wires it.

The tee holds a *real* primary (#335): the level-aware console, or the quiet
:data:`~noctis.observability.NULL_SINK` when a ``--debug`` run records with no ``-v``. So every
event reaches the primary and every attribute delegates to it, unconditionally — that the whole
seven-member surface survives the delegation is the sink-contract test's job
(``EventTee(NULL_SINK)`` is one of its adapters). What is pinned *here* is what the tee adds:
events reach the primary **and** every secondary, a raising secondary is isolated from both, a
real console's state reads back through, and the builder wires the right shape.
"""

from __future__ import annotations

from noctis.observability import NULL_SINK, Console, Event, EventTee, NullSink
from noctis.observability import events as events_module
from noctis.observability import tee as tee_module


def _console(verbose=2, **kw):
    """A real Console whose block output lands in a list — the tee delegates to *this*, so the
    delegation test asserts against a genuine console surface, not a stand-in."""
    out: list[str] = []
    kw.setdefault("color", False)
    return Console(verbose, sink=out.append, **kw), out


# ── one event, both destinations ──────────────────────────────────────────────────────────────
def test_event_reaches_both_primary_and_secondary():
    con, out = _console(verbose=1)
    recorded: list = []
    tee = EventTee(con, recorded.append)
    ev = Event("tool", "x -> ok", level=1)
    tee(ev)
    assert out == ["→ x -> ok"]  # rendered on the console
    assert recorded == [ev]  # and recorded on the secondary


def test_events_still_reach_secondaries_when_the_primary_is_quiet():
    """A quiet ``--debug`` run: nothing is watching, so the primary is the null sink and the call
    on it is inert — but the recorder still gets every event."""
    recorded: list = []
    tee = EventTee(NULL_SINK, recorded.append)
    ev = Event("tool", "run_backtest(...) -> ok", level=1)
    tee(ev)
    assert recorded == [ev]


# ── a raising secondary never breaks the primary path ─────────────────────────────────────────
def test_raising_secondary_does_not_break_the_primary_console():
    con, out = _console(verbose=1)

    def boom(_ev):
        raise RuntimeError("recorder blew up")

    tee = EventTee(con, boom)
    tee(Event("tool", "x -> ok", level=1))  # must not propagate the recorder's failure
    assert out == ["→ x -> ok"]  # the primary console rendered regardless


def test_raising_secondary_does_not_break_later_secondaries():
    later: list = []

    def boom(_ev):
        raise RuntimeError("first recorder blew up")

    tee = EventTee(NULL_SINK, boom, later.append)
    ev = Event("tool", "x -> ok", level=1)
    tee(ev)
    assert later == [ev]  # a raising secondary is isolated; the next one still receives the event


# ── delegation: whatever primary the tee holds is the one it reads ────────────────────────────
def test_a_real_consoles_state_reads_through_the_tee():
    """The agent loop gates streaming on ``on_event.verbose >= 2`` and the CLI reads ``saw_think``
    afterwards — both off the tee, both answered by the console the tee is fronting."""
    con, _out = _console(verbose=2)
    tee = EventTee(con, [].append)
    assert tee.verbose == 2
    assert tee.saw_think is False  # nothing surfaced yet
    tee(Event("think", "reasoning", level=2))  # a think event flows through the tee to the console
    assert tee.saw_think is True  # the console flipped its flag; the tee reads it through


def test_delegation_reads_through_a_primary_that_is_falsy():
    """The tee delegates to the primary it *holds*, not to whichever stand-in an ``or`` picks.
    A sink can read as falsy — a recorder that reports how many events it has filed is empty at
    the start of a run — and it is still the primary, so its verbosity is what a caller reads
    off the tee."""

    class CountingSink(NullSink):
        verbose = 2

        def __len__(self) -> int:
            return 0  # falsy until it has filed its first event

    tee = EventTee(CountingSink(), [].append)
    assert tee.verbose == 2


def test_the_tee_names_the_shared_sink_protocol():
    """One sink contract, one null adapter, for the whole package: the tee's ``EventSink`` *is*
    the events module's Protocol (not a one-member callable alias of its own), and the private
    null console it used to fall back on is gone — the quiet primary is ``NULL_SINK``."""
    assert tee_module.EventSink is events_module.EventSink
    assert not hasattr(tee_module, "_NullConsole")


# ── the event-sink builder ────────────────────────────────────────────────────────────────────
def test_build_event_sink_without_secondary_is_the_console_itself():
    from noctis.bootstrap import build_event_sink

    assert build_event_sink(0) is None  # no secondary, quiet ⇒ exactly the old None
    con = build_event_sink(1)
    assert isinstance(con, Console) and not isinstance(con, EventTee)  # a bare console, no tee


def test_build_event_sink_with_secondary_records_on_a_quiet_run():
    from noctis.bootstrap import build_event_sink

    recorded: list = []
    sink = build_event_sink(0, secondary=recorded.append)  # quiet --debug: no console, still record
    assert isinstance(sink, EventTee)
    assert sink.verbose == 0  # no console ⇒ the null sink's quiet default reads through
    ev = Event("tool", "x -> ok", level=1)
    sink(ev)
    assert recorded == [ev]  # the recorder gets the event even with no -v console


def test_build_event_sink_with_secondary_and_verbose_tees_to_both():
    from noctis.bootstrap import build_event_sink

    recorded: list = []
    sink = build_event_sink(1, secondary=recorded.append)
    assert isinstance(sink, EventTee)
    assert sink.verbose == 1  # a real console primary is present and its verbosity reads through
    ev = Event("tool", "x -> ok", level=1)
    sink(ev)
    assert recorded == [ev]  # the recorder still gets a copy alongside the console
