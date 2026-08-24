"""The event sink's contract, written down once and enforced over every adapter (#334, epic #333).

A session hands its observable moments to one ``on_event`` sink, and callers duck-type **seven**
members off whatever object arrives: the call itself (an :class:`Event` or a legacy pre-formatted
string), ``delta``/``hint``/``activity``, and the ``verbose``/``show_reasoning``/``saw_think``
reads the agent loop and the CLI make. :class:`~noctis.observability.events.EventSink` is that
surface as a type, and :class:`~noctis.observability.events.NullSink` is its safe-default adapter.

These tests are the contract. The conformance test is **parametrized over adapters** — the
renderer (``Console``), the quiet default (``NullSink``), the splitter (``EventTee``, #335) and
the ``--debug`` ``Recorder`` (#336), which declares the seam by subclassing ``NullSink`` and so
joins :data:`ADAPTERS` as one entry rather than bringing a second set of by-name tests with it.
Everything asserted here is what a caller observes: what the sink accepts, what it returns, that
``activity`` brackets a block, and what the three flags read as.
"""

from __future__ import annotations

import ast
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from noctis.observability import NULL_SINK, Console, Event, EventSink, EventTee, NullSink
from noctis.observability import events as events_module
from noctis.observability.debug import Recorder

# The seam's whole surface, named once. Every adapter serves all seven; dropping any one of them
# breaks conformance (that is the point of writing the contract down).
SINK_MEMBERS = (
    "__call__",
    "delta",
    "hint",
    "activity",
    "verbose",
    "show_reasoning",
    "saw_think",
)


# Every builder takes the test's throwaway directory, because one adapter — the recorder — files
# to disk; the three in-memory adapters ignore it.
def _console(qa_dir: Path) -> Console:
    """A real :class:`Console` whose output lands in a list, off a TTY — the adapter as the CLI
    builds it, minus the terminal."""
    return Console(2, color=False, sink=[].append, tty=False)


def _null_sink(qa_dir: Path) -> NullSink:
    """The quiet default — stateless, so it needs nothing built for it."""
    return NullSink()


def _tee(qa_dir: Path) -> EventTee:
    """A tee fronting the quiet null sink — the shape a ``--debug`` run with no ``-v`` builds. It
    is a sink because it delegates every read to the real primary it holds (#335)."""
    return EventTee(NULL_SINK, [].append)


def _recorder(qa_dir: Path) -> Recorder:
    """The ``--debug`` recorder filing into a throwaway QA tree on a frozen clock — the adapter as
    ``build_recorder`` assembles it, minus the run. It serves the seam by subclassing
    :class:`NullSink`: its own call records, and the six console-facing members it never
    implemented are the null adapter's safe defaults (#336)."""
    return Recorder(qa_dir, run_id="conformance", clock=lambda: datetime(2026, 7, 20, 14, 0))


# The adapters that declare the seam — all four of them, each one entry here rather than a new set
# of by-name tests.
ADAPTERS = {
    "console": _console,
    "null-sink": _null_sink,
    "event-tee": _tee,
    "recorder": _recorder,
}


# ── the conformance test: every adapter serves all seven members ─────────────────────────────
@pytest.mark.parametrize("build_adapter", list(ADAPTERS.values()), ids=list(ADAPTERS))
def test_every_adapter_serves_the_whole_sink_surface(build_adapter, tmp_path):
    sink = build_adapter(tmp_path)
    assert isinstance(sink, EventSink)
    # 1-2. The call takes a typed Event and a legacy pre-formatted string alike.
    assert sink(Event("tool", "run_backtest(...) -> ok", level=1)) is None
    assert sink("web_search disabled for this session") is None
    # 3-4. The streaming delta and the advisory hint are plain calls that return nothing.
    assert sink.delta("think", "streamed reasoning") is None
    assert sink.hint("reasoning not surfaced") is None
    # 5. activity() brackets a blocking call as a context manager.
    entered = False
    with sink.activity("optimize donchian_breakout"):
        entered = True
    assert entered
    # 6-7. The three flags the agent loop and the CLI read off the sink.
    assert isinstance(sink.verbose, int)
    assert isinstance(sink.show_reasoning, bool)
    assert isinstance(sink.saw_think, bool)


# ── the protocol demands exactly those seven ─────────────────────────────────────────────────
@contextmanager
def _activity(self, label):
    yield


# A minimal adapter built from nothing but the seven members — the smallest thing that is a sink.
_SEVEN_MEMBERS = {
    "__call__": lambda self, ev: None,
    "delta": lambda self, kind, text: None,
    "hint": lambda self, text: None,
    "activity": _activity,
    "verbose": 0,
    "show_reasoning": False,
    "saw_think": False,
}


def _adapter(*, without: str = "") -> object:
    """An instance of a hand-rolled adapter carrying the seven members, minus ``without``."""
    members = {name: m for name, m in _SEVEN_MEMBERS.items() if name != without}
    return type("HandRolledSink", (), members)()


def test_an_adapter_with_exactly_the_seven_members_is_an_event_sink():
    assert isinstance(_adapter(), EventSink)


@pytest.mark.parametrize("missing", SINK_MEMBERS)
def test_an_adapter_missing_any_member_is_not_an_event_sink(missing):
    assert not isinstance(_adapter(without=missing), EventSink)


def test_a_bare_callable_is_not_an_event_sink():
    """The old declared type was ``Callable[[Event | str], None]`` — one member of seven. A list's
    ``append`` still rides the tee as a secondary, but it is not the seam itself."""
    assert not isinstance([].append, EventSink)


# ── the null adapter's safe defaults ─────────────────────────────────────────────────────────
def test_the_null_sink_swallows_an_event_and_a_legacy_line():
    """``NullSink.__call__`` exists — the private stand-in it replaces had none, because the tee
    never called it; from story #335 the tee always calls its primary."""
    sink = NullSink()
    assert sink(Event("think", "reasoning", level=2)) is None
    assert sink("web_search disabled for this session") is None


def test_the_null_sink_reads_as_a_quiet_console():
    sink = NullSink()
    assert sink.verbose == 0  # parks the agent loop's `verbose >= 2` streaming gate
    assert sink.show_reasoning is False
    assert sink.saw_think is False


def test_the_null_sink_yields_from_activity_and_stays_inert():
    sink = NullSink()
    assert sink.delta("think", "streamed reasoning") is None
    assert sink.hint("reasoning not surfaced") is None
    entered = False
    with sink.activity("model call"):
        entered = True
    assert entered  # the no-op context manager yielded normally


def test_an_unknown_duck_typed_read_on_the_null_sink_is_an_inert_call():
    """A future caller reaching for a member the seam does not have gets a harmless no-op, the
    same tail the private stand-in carried — nothing a quiet run touches ever raises."""
    assert NullSink().some_future_callback("x") is None


def test_the_null_sink_is_stateless_so_one_singleton_serves_every_session():
    assert isinstance(NULL_SINK, NullSink)
    NULL_SINK(Event("think", "reasoning", level=2))  # a console would flip saw_think here
    with NULL_SINK.activity("model call"):
        NULL_SINK.delta("think", "streamed reasoning")
    assert (NULL_SINK.verbose, NULL_SINK.show_reasoning, NULL_SINK.saw_think) == (0, False, False)


# ── the package surface ──────────────────────────────────────────────────────────────────────
def test_the_package_re_exports_the_contract_from_the_events_module():
    import noctis.observability as observability

    assert observability.EventSink is events_module.EventSink
    assert observability.NullSink is events_module.NullSink
    assert observability.NULL_SINK is events_module.NULL_SINK
    assert {"EventSink", "NullSink", "NULL_SINK"} <= set(observability.__all__)


# ── the events module stays core-only ────────────────────────────────────────────────────────
def _import_roots(source: str) -> set[str]:
    """The top-level package every import in ``source`` names — module level and inside a function
    body alike, since a deferred import is still a dependency."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_events_module_imports_nothing_but_the_standard_library():
    """The contract lives where every backend can reach it: no console, no tee, no recorder, and
    no heavy package — so declaring a sink costs a renderer's import, and a core install nothing."""
    source = Path(events_module.__file__).read_text(encoding="utf-8")
    assert _import_roots(source) - set(sys.stdlib_module_names) == set()
