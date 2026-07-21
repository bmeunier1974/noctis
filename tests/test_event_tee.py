"""The observability tee (epic #36): an :class:`EventTee` that rides a recorder alongside the
console on the one ``on_event`` seam, plus the event-sink builder that wires it.

Callers duck-type the console sink — a ``delta``/``activity`` pair, a ``hint`` line, and the
``verbose``/``show_reasoning``/``saw_think`` attributes the agent loop reads to decide whether to
stream — so the tee must forward events to a recorder *and* keep that whole surface working. With
no ``-v`` console the primary is absent; delegation then has to resolve to safe no-ops, never
raise. These tests pin every proxied attribute by name, the no-primary case, the guard around a
raising secondary, and the builder's back-compat.
"""

from __future__ import annotations

from noctis.observability import Console, Event, EventTee


def _console(verbose=2, **kw):
    """A real Console whose block output lands in a list — the tee delegates to *this*, so the
    proxy tests assert against a genuine console surface, not a stand-in."""
    out: list[str] = []
    kw.setdefault("color", False)
    return Console(verbose, sink=out.append, **kw), out


# ── each proxied attribute, by name (AC 1) ────────────────────────────────────────────────────
def test_reasoning_delta_reaches_the_primary_console():
    con, _out = _console(tty=True, stream_sink=(stream := []).append)
    tee = EventTee(con, [].append)
    tee.delta("think", "streamed reasoning")
    assert "streamed reasoning" in "".join(stream)  # the delta rendered in place on the primary


def test_activity_context_manager_reaches_the_primary_console():
    con, out = _console(verbose=1, tty=False)  # off a TTY: one breadcrumb, no spinner thread
    tee = EventTee(con, [].append)
    with tee.activity("optimize donchian_breakout"):
        pass
    assert out == ["⠿ optimize donchian_breakout …"]  # the primary's activity line, via the tee


def test_hint_reaches_the_primary_console():
    con, out = _console(verbose=1)
    tee = EventTee(con, [].append)
    tee.hint("reasoning not surfaced")
    assert out == ["(reasoning not surfaced)"]  # the primary's dim advisory, via the tee


def test_verbosity_reads_through_from_the_primary():
    con, _out = _console(verbose=2)
    tee = EventTee(con, [].append)
    assert tee.verbose == 2  # the agent loop gates streaming on `on_event.verbose >= 2`


def test_saw_think_reads_through_from_the_primary():
    con, _out = _console(verbose=2)
    tee = EventTee(con, [].append)
    assert tee.saw_think is False  # nothing surfaced yet
    tee(Event("think", "reasoning", level=2))  # a think event flows through the tee to the console
    assert tee.saw_think is True  # the console flipped its flag; the tee reads it through


# ── no-primary case is safe (AC 2) ────────────────────────────────────────────────────────────
def test_no_primary_attribute_access_never_raises():
    recorded: list = []
    tee = EventTee(None, recorded.append)
    # Attribute reads resolve to safe defaults instead of raising.
    assert tee.verbose == 0
    assert tee.show_reasoning is False
    assert tee.saw_think is False
    # Method-style callbacks are inert no-ops (no console to render to).
    assert tee.delta("think", "hi") is None
    assert tee.hint("note") is None
    with tee.activity("optimize foo"):  # still usable as a context manager
        pass
    # Even an unknown duck-typed attribute resolves to a harmless no-op callable.
    assert tee.some_future_callback("x") is None


def test_no_primary_events_still_reach_secondaries():
    recorded: list = []
    tee = EventTee(None, recorded.append)
    ev = Event("tool", "run_backtest(...) -> ok", level=1)
    tee(ev)
    assert recorded == [ev]  # the recorder still gets every event with no console present


# ── a raising secondary never breaks the primary path (AC 3) ──────────────────────────────────
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

    tee = EventTee(None, boom, later.append)
    ev = Event("tool", "x -> ok", level=1)
    tee(ev)
    assert later == [ev]  # a raising secondary is isolated; the next one still receives the event


def test_event_reaches_both_primary_and_secondary():
    con, out = _console(verbose=1)
    recorded: list = []
    tee = EventTee(con, recorded.append)
    ev = Event("tool", "x -> ok", level=1)
    tee(ev)
    assert out == ["→ x -> ok"]  # rendered on the console
    assert recorded == [ev]  # and recorded on the secondary


# ── the event-sink builder (AC 4) ─────────────────────────────────────────────────────────────
def test_build_console_alias_is_unchanged():
    from noctis.bootstrap import build_console

    assert build_console(0) is None  # quiet run ⇒ None ⇒ loops stay on their logger sinks
    assert isinstance(build_console(1), Console)  # -v ⇒ a level-aware console


def test_build_event_sink_without_secondary_matches_build_console():
    from noctis.bootstrap import build_event_sink

    assert build_event_sink(0) is None  # no secondary, quiet ⇒ exactly the old None
    con = build_event_sink(1)
    assert isinstance(con, Console) and not isinstance(con, EventTee)  # a bare console, no tee


def test_build_event_sink_with_secondary_records_on_a_quiet_run():
    from noctis.bootstrap import build_event_sink

    recorded: list = []
    sink = build_event_sink(0, secondary=recorded.append)  # quiet --debug: no console, still record
    assert isinstance(sink, EventTee)
    assert sink.verbose == 0  # no primary ⇒ the safe default reads through
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


# ── the null-console fallback delegates unknown reads to a no-op too ───────────────────────────
def test_activity_no_primary_is_a_working_context_manager():
    """The agent loop calls ``on_event.activity(label)`` and enters the result; with no console
    the tee must still hand back something ``with``-usable, or a quiet --debug run would crash."""
    tee = EventTee(None, [].append)
    entered = False
    with tee.activity("model call"):
        entered = True
    assert entered  # the no-op context manager yielded normally
